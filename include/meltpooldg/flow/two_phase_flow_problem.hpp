/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, Peter Münch, TUM, October 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria_base.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/grid/grid_out.h>
// MeltPoolDG
#include <meltpooldg/flow/flow_base.hpp>
#include <meltpooldg/flow/adaflo_wrapper.hpp>
#include <meltpooldg/interface/problembase.hpp>
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/level_set/level_set_operation.hpp>
#include <meltpooldg/utilities/timeiterator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG
{
  namespace Flow
  {
    using namespace dealii;

    template <int dim>
    class TwoPhaseFlowProblem : public ProblemBase<dim>
    {
    private:
      using VectorType      = LinearAlgebra::distributed::Vector<double>;
      using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    public:
      TwoPhaseFlowProblem() = default;

      void
      run(std::shared_ptr<SimulationBase<dim>> base_in) final
      {
        initialize(base_in);

        output_results(0,base_in->parameters);
        
        while (!time_iterator.is_finished())
          {
            const auto dt = time_iterator.get_next_time_increment();
            const auto n  = time_iterator.get_current_time_step_number();

            scratch_data->get_pcout()
              << "t= " << std::setw(10) << std::left << time_iterator.get_current_time();

            // ... solve level-set problem with the given advection field
            level_set_operation.solve(dt, advection_velocity);
            
            // update 
            update_phases(level_set_operation.solution_level_set, base_in->parameters);
            
            // accumulate forces: a) gravity force
            compute_gravity_force(force_rhs, base_in->parameters.base.gravity); 
            // ... b) surface tension
            level_set_operation.compute_surface_tension(
              force_rhs, base_in->parameters.flow.surface_tension_coefficient, true /*add to force vector*/);
            
            //  ... and set forces within the Navier-Stokes solver
            flow_operation->set_force_rhs(force_rhs);

            // solver Navier-Stokes problem
            flow_operation->solve();

            // extract velocity form Navier-Stokes solver ...
            flow_operation->get_velocity(advection_velocity);

            output_results(n, base_in->parameters);
          }
      }

      std::string
      get_name() final
      {
        return "two_phase_flow";
      };

    private:
      /*
       *  This function initials the relevant scratch data
       *  for the computation of the level set problem
       */
      void
      initialize(std::shared_ptr<SimulationBase<dim>> base_in)
      {
        /*
         *  setup scratch data
         */
        scratch_data = std::make_shared<ScratchData<dim>>(/*do_matrix_free*/ true);
        /*
         *  setup mapping
         */
        scratch_data->set_mapping(MappingQGeneric<dim>(base_in->parameters.base.degree));
        /*
         *  setup DoFHandler
         */
        dof_handler.initialize(*base_in->triangulation, FE_Q<dim>(base_in->parameters.base.degree));

        scratch_data->attach_dof_handler(dof_handler);
        scratch_data->attach_dof_handler(dof_handler);
        /*
         *  create partitioning
         */
        scratch_data->create_partitioning();
        /*
         *  make hanging nodes and dirichlet constraints (at the moment no time-dependent
         *  dirichlet constraints are supported)
         */
        hanging_node_constraints.clear();
        hanging_node_constraints.reinit(scratch_data->get_locally_relevant_dofs());
        DoFTools::make_hanging_node_constraints(dof_handler, hanging_node_constraints);
        hanging_node_constraints.close();

        constraints_dirichlet.clear();
        constraints_dirichlet.reinit(scratch_data->get_locally_relevant_dofs());
        for (const auto &bc : base_in->get_dirichlet_bc("level_set")) // @todo: add name of bc at a more central place
          {
            dealii::VectorTools::interpolate_boundary_values(scratch_data->get_mapping(),
                                                             dof_handler,
                                                             bc.first,
                                                             *bc.second,
                                                             constraints_dirichlet);
          }
        constraints_dirichlet.merge(hanging_node_constraints);
        constraints_dirichlet.close();

        dof_no_bc_idx =
          scratch_data->attach_constraint_matrix(hanging_node_constraints);
        dof_idx = scratch_data->attach_constraint_matrix(constraints_dirichlet);
        /*
         *  create quadrature rule
         */
        quad_idx =
          scratch_data->attach_quadrature(QGauss<1>(base_in->parameters.base.n_q_points_1d));
        /*
         *  create the matrix-free object
         */
        scratch_data->build();
        /*
         *  initialize the time stepping scheme
         */
        time_iterator.initialize(
          TimeIteratorData<double>{base_in->parameters.flow.start_time,
                                   base_in->parameters.flow.end_time,
                                   base_in->parameters.flow.time_step_size,
                                   base_in->parameters.flow.max_n_steps,
                                   false /*cfl_condition-->not supported yet*/});
        /*
         *  initialize the velocity dof vector
         */
        scratch_data->initialize_dof_vector(advection_velocity, dof_no_bc_idx);
        /*
         *  set initial conditions of the levelset function
         */
         AssertThrow(base_in->get_initial_condition("level_set"),
         ExcMessage(
           "It seems that your SimulationBase object does not contain "
           "a valid initial field function for the level set field. A shared_ptr to your initial field "
           "function, e.g., MyInitializeFunc<dim> must be specified as follows: "
           "  this->attach_initial_condition(std::make_shared<MyInitializeFunc<dim>>(), 'level_set') "));

        VectorType initial_solution;
        scratch_data->initialize_dof_vector(initial_solution, dof_idx);
        dealii::VectorTools::project(scratch_data->get_mapping(),
                                     dof_handler,
                                     constraints_dirichlet,
                                     scratch_data->get_quadrature(),
                                    *base_in->get_initial_condition("level_set"),
                                     initial_solution);
        initial_solution.update_ghost_values();
        /*
         *    initialize the levelset operation class
         */
        level_set_operation.initialize(
          scratch_data, initial_solution, base_in->parameters, dof_idx, dof_no_bc_idx, quad_idx);

#ifdef MELT_POOL_DG_WITH_ADAFLO
        flow_operation = std::make_shared<AdafloWrapper<dim>>(*scratch_data,
                                                              dof_idx,
                                                              base_in);
#else
        AssertThrow(false, ExcNotImplemented ());
#endif
       /*
         *    initialize the force vector for calculating surface tension
         */
        scratch_data->initialize_dof_vector(force_rhs, dof_no_bc_idx);  
       /*
         *    initialize the density and viscosity vector (for postprocessing)
         */
        scratch_data->initialize_dof_vector(density, dof_no_bc_idx);  
        scratch_data->initialize_dof_vector(viscosity, dof_no_bc_idx);  

      }

      /**
       * Update material parameter of the phases.
       * 
       * @todo Find a better place.
       */
      void
      update_phases(const VectorType & src, const Parameters<double> &parameters) const
      {
        double dummy;

      scratch_data->get_matrix_free().template cell_loop<double, VectorType>(
        [&](const auto &matrix_free, auto &, const auto &src, auto macro_cells) {
          
          FECellIntegrator<dim, 1, double> ls_values(matrix_free, dof_idx, quad_idx);

          for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
            {
              ls_values.reinit(cell);
              ls_values.read_dof_values_plain(src);
              ls_values.evaluate(true, false);
              
              for (unsigned int q = 0; q < ls_values.n_q_points; ++q)
                {
                  // convert level-set value to heaviside function 
                  const auto indicator = UtilityFunctions::heaviside(ls_values.get_value(q),0.0);
                  
                  // set density
                  flow_operation->get_density(cell, q) = parameters.flow.density + parameters.flow.density_difference * indicator;

                  // set viscosity
                  flow_operation->get_viscosity(cell, q) = parameters.flow.viscosity + parameters.flow.viscosity_difference * indicator;
                }
            }
        },
        dummy,
        src);
      }

      /**
       * Compute gravity force.
       * 
       * @todo Find a better place.
       */
      void
      compute_gravity_force(BlockVectorType & vec, const double gravity, const bool add = false) const
      {
        scratch_data->get_matrix_free().template cell_loop<BlockVectorType, std::nullptr_t>(
          [&](const auto &matrix_free, auto & vec, const auto &, auto macro_cells) {
            
            FECellIntegrator<dim, dim, double> force_values(matrix_free, dof_no_bc_idx, quad_idx);
  
            for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
              {
                force_values.reinit(cell);
                
                for (unsigned int q = 0; q < force_values.n_q_points; ++q)
                  {
                    Tensor<1, dim, VectorizedArray<double> > force;
                    
                    force[dim - 1] -= gravity * flow_operation->get_density(cell, q);
                    force_values.submit_value(force, q);
                  }
                
                force_values.integrate_scatter(true, false, vec);
              }
          },
          vec,
          nullptr,
          !add);
      }

      void
      create_density_dof_vector(VectorType & vec) const
      {
        (void)vec;
        //@todo: does not compile
        //dealii::VectorTools::project(std::make_shared<MatrixFree<dim,double,VectorizedArray<double>>(scratch_data->get_matrix_free()),
                                     //scratch_data->get_constraint(dof_no_bc_idx),
                                     //1,
                                      //[&] (const unsigned int cell,
                                           //const unsigned int q) -> VectorizedArray<double>
                                     //{ 
                                     //return  flow_operation->get_density(cell, q); 
                                     //},
                                     //vec);
      }
      
      void
      create_viscosity_dof_vector(VectorType & vec) const
      {
        (void)vec;
      }
      
      /*
       *  This function is to create paraview output
       */
      void
      output_results(const unsigned int time_step, const Parameters<double> &parameters) 
      {
        
        if (parameters.paraview.do_output)
        {
          create_density_dof_vector(density);
          create_viscosity_dof_vector(viscosity);
          // update ghost values
          VectorTools::update_ghost_values( advection_velocity,
                                            force_rhs,
                                            density,
                                            viscosity,
                                            level_set_operation.solution_level_set,
                                            level_set_operation.solution_curvature,
                                            level_set_operation.solution_normal_vector);


          /*
           *  output advected field
           */
          DataOut<dim> data_out;
          data_out.attach_dof_handler(dof_handler);

          for (auto d = 0; d < dim; ++d)
            data_out.add_data_vector(dof_handler,
                                     advection_velocity.block(d),
                                     "advection_velocity_" + std::to_string(d));

          /*
           * level set
           */
          data_out.add_data_vector(level_set_operation.solution_level_set, "level_set");
          /*
           * curvature
           */
          
          data_out.add_data_vector(level_set_operation.solution_curvature, "curvature");
          /*
           *  normal vector field
           */
           for (unsigned int d = 0; d < dim; ++d)
              data_out.add_data_vector(level_set_operation.solution_normal_vector.block(d),
                                       "normal_" + std::to_string(d));

          /*
           * force vector (surface tension + gravity force)
           */
           for (auto d = 0; d < dim; ++d)
             data_out.add_data_vector(dof_handler,
                                      force_rhs.block(d),
                                     "force_" + std::to_string(d));
          /*
           * density
           */
          data_out.add_data_vector(dof_handler,
                                   density,
                                   "density");
          /*
           * viscosity 
           */
          data_out.add_data_vector(dof_handler,
                                   viscosity,
                                   "viscosity");

          data_out.build_patches(scratch_data->get_mapping());
          data_out.write_vtu_with_pvtu_record("./",
                                              parameters.paraview.filename,
                                              time_step,
                                              scratch_data->get_mpi_comm(),
                                              parameters.paraview.n_digits_timestep,
                                              parameters.paraview.n_groups);

        }

        // clear ghost values
        VectorTools::zero_out_ghosts( advection_velocity,
                                      force_rhs,
                                      density,
                                      viscosity,
                                      level_set_operation.solution_level_set,
                                      level_set_operation.solution_curvature,
                                      level_set_operation.solution_normal_vector );
      }

    private:
      TimeIterator<double> time_iterator;
      DoFHandler<dim>      dof_handler;

      AffineConstraints<double> constraints_dirichlet;
      AffineConstraints<double> hanging_node_constraints;

      BlockVectorType advection_velocity;
      BlockVectorType force_rhs;
      VectorType      density;
      VectorType      viscosity;

      unsigned int dof_idx;
      unsigned int dof_no_bc_idx;
      unsigned int quad_idx;
      
      std::shared_ptr<ScratchData<dim>> scratch_data;
      std::shared_ptr<FlowBase>         flow_operation;
      LevelSet::LevelSetOperation<dim>  level_set_operation;
    };
  } // namespace Flow
} // namespace MeltPoolDG
