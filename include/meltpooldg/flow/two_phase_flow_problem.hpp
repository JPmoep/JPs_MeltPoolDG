/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
// enabling conditional ostreams
#include <deal.II/base/conditional_ostream.h>
// for index set
#include <deal.II/base/index_set.h>
//// for distributed triangulation
#include <deal.II/distributed/tria_base.h>
// for dof_handler type
#include <deal.II/dofs/dof_handler.h>
// for data_out
#include <deal.II/base/data_out_base.h>

#include <deal.II/numerics/data_out.h>
// for FE_Q<dim> type
#include <deal.II/fe/fe_q.h>
// for mapping
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_out.h>
// MeltPoolDG
#include <meltpooldg/flow/adaflo_wrapper.hpp>
#include <meltpooldg/interface/problembase.hpp>
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/level_set/level_set_operation.hpp>
#include <meltpooldg/utilities/timeiterator.hpp>


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

        // TODO: make class field?
        BlockVectorType surface_tension_force;
        scratch_data->initialize_dof_vector(surface_tension_force, dof_idx);

        // initialize phases and force(?) [TODO] 
        update_phases(level_set_operation.solution_level_set, base_in->parameters);
        
        output_results(0, base_in->parameters);
        
        // TODO: re-enable?
        // output_results(0,base_in->parameters);
        while (!time_iterator.is_finished())
          {
            const auto dt = time_iterator.get_next_time_increment();
            const auto n  = time_iterator.get_current_time_step_number();

            // solver Navier-Stokes problem
            flow_operation->solve();

            // extract velocity form Navier-Stokes solver ...
            flow_operation->get_velocity(advection_velocity);

            // ... solve level-set problem with the given advection field
            level_set_operation.solve(dt, advection_velocity);
            
            // update 
            update_phases(level_set_operation.solution_level_set, base_in->parameters);
            
            // accumulate forces: a) gravity force
            compute_gravity_force(surface_tension_force, false);
            
            // ... b) surface tension
            level_set_operation.compute_surface_tension(
              surface_tension_force, base_in->parameters.flow.surface_tension_coefficient, true);

            //  ... and set forces within the Navier-Stokes solver
            flow_operation->set_surface_tension(surface_tension_force);

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
        constraints_dirichlet.merge(hanging_node_constraints);
        for (const auto &bc : base_in->get_dirichlet_bc("level_set")) // @todo: add name of bc at a more central place
          {
            dealii::VectorTools::interpolate_boundary_values(scratch_data->get_mapping(),
                                                             dof_handler,
                                                             bc.first,
                                                             *bc.second,
                                                             constraints_dirichlet);
          }
        constraints_dirichlet.close();

        // scratch_data->attach_constraint_matrix(dummy_constraint);
        const unsigned int dof_no_bc_idx =
          scratch_data->attach_constraint_matrix(hanging_node_constraints);
        dof_idx = scratch_data->attach_constraint_matrix(constraints_dirichlet);
        /*
         *  create quadrature rule
         */
        unsigned int quad_idx =
          scratch_data->attach_quadrature(QGauss<1>(base_in->parameters.base.n_q_points_1d));
        /*
         *  create the matrix-free object
         */
        scratch_data->build();

        time_iterator.initialize(
          TimeIteratorData<double>{base_in->parameters.flow.start_time,
                                   base_in->parameters.flow.end_time,
                                   base_in->parameters.flow.time_step_size,
                                   base_in->parameters.flow.max_n_steps,
                                   false /*cfl_condition-->not supported yet*/});

        scratch_data->initialize_dof_vector(advection_velocity, dof_idx);
        /*
         *  set initial conditions of the levelset function
         */
        VectorType initial_solution;
        scratch_data->initialize_dof_vector(initial_solution);
        dealii::VectorTools::project(scratch_data->get_mapping(),
                                     dof_handler,
                                     constraints_dirichlet,
                                     scratch_data->get_quadrature(),
                                     *base_in->get_field_conditions()->initial_field,
                                     initial_solution);

        initial_solution.update_ghost_values();
        /*
         *    initialize the levelset operation class
         */


        level_set_operation.initialize(
          scratch_data, initial_solution, base_in->parameters, dof_idx, dof_no_bc_idx, quad_idx);

        flow_operation = std::make_shared<AdafloWrapper<dim>>(*scratch_data,
                                                              dof_idx,
                                                              base_in);
      }

      /**
       * Update material parameter of the phases.
       * 
       * @todo Find a better place.
       */
      void
      update_phases(const VectorType & vec, const Parameters<double> &parameters) const
      {
        double dummy;
          
      scratch_data->get_matrix_free().template cell_loop<double, VectorType>(
        [&](const auto &matrix_free, auto &, const auto &src, auto macro_cells) {
          
          FECellIntegrator<dim, 1, double> ls_values(matrix_free, 0 /*TODO*/, 0 /*TODO*/);

          for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
            {
              ls_values.reinit(cell);
              ls_values.read_dof_values_plain(src);
              ls_values.evaluate(true, false);
              
              for (unsigned int q = 0; q < ls_values.n_q_points; ++q)
                {
                  // convert level-set value to heaviside function
                  const auto indicator = (ls_values.get_value(q) + 1.0) * 0.5;
                  
                  // set density
                  flow_operation->get_density(cell, q) = parameters.flow.density + parameters.flow.density_difference * indicator;
                  
                  // set viscosity
                  flow_operation->get_viscosity(cell, q) = parameters.flow.viscosity + parameters.flow.viscosity_difference * indicator;
                }
            }
        },
        dummy,
        vec);
      }

      /**
       * Compute gravity force.
       * 
       * @todo Find a better place.
       */
      void
      compute_gravity_force(BlockVectorType & vec, const bool add = true) const
      {
          
        const double gravity = 0.00; // TODO
          
        scratch_data->get_matrix_free().template cell_loop<BlockVectorType, std::nullptr_t>(
          [&](const auto &matrix_free, auto & force_rhs, const auto &, auto macro_cells) {
            
            FECellIntegrator<dim, dim, double> force_values(matrix_free, 0 /*TODO*/, 0 /*TODO*/);
  
            for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
              {
                force_values.reinit(cell);
                
                for (unsigned int q = 0; q < force_values.n_q_points; ++q)
                  {
                    Tensor<1, dim, VectorizedArray<double> > force;
                    
                    force[dim - 1] -= gravity * flow_operation->get_density(cell, q);
                    
                    force_values.submit_value(force, q);
                  }
                
                force_values.integrate_scatter(true, false, force_rhs);
              }
          },
          vec,
          nullptr,
          add);
      }
      
      /*
       *  This function is to create paraview output
       */
      void
      output_results(const unsigned int time_step, const Parameters<double> &parameters) const
      {
        // update ghost values
        advection_velocity.update_ghost_values();

        // if (parameters.paraview.do_output)
        // {
        /*
         *  output advected field
         */
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);

        for (auto d = 0; d < dim; ++d)
          data_out.add_data_vector(dof_handler,
                                   advection_velocity.block(d),
                                   "advection_velocity_" + std::to_string(d));

        data_out.add_data_vector(level_set_operation.solution_level_set, "level_set");

        /*
         * plot surface-tension
         */
        // for (auto d = 0; d < dim; ++d)
        //   data_out.add_data_vector(dof_handler,
        //                            surface_tension_force.block(d),
        //                           "surface_tension_force_" + std::to_string(d));

        data_out.build_patches(scratch_data->get_mapping());
        data_out.write_vtu_with_pvtu_record("./",
                                            parameters.paraview.filename,
                                            time_step,
                                            scratch_data->get_mpi_comm(),
                                            parameters.paraview.n_digits_timestep,
                                            parameters.paraview.n_groups);

        // }

        // clear ghost values
        advection_velocity.zero_out_ghosts();
      }

    private:
      TimeIterator<double> time_iterator;
      DoFHandler<dim>      dof_handler;

      AffineConstraints<double> constraints_dirichlet;
      AffineConstraints<double> hanging_node_constraints;
      AffineConstraints<double> dummy_constraints;

      BlockVectorType advection_velocity;

      unsigned int                      dof_idx;
      std::shared_ptr<ScratchData<dim>> scratch_data;
      std::shared_ptr<FlowBase>         flow_operation;
      LevelSet::LevelSetOperation<dim>  level_set_operation;
    };
  } // namespace Flow
} // namespace MeltPoolDG
