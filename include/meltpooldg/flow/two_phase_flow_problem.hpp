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
#include <meltpooldg/interface/problembase.hpp>
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/level_set/level_set_operation.hpp>
#include <meltpooldg/utilities/timeiterator.hpp>
#include <meltpooldg/flow/adaflo_wrapper.hpp>
#include <meltpooldg/flow/adaflo_wrapper_parameters.hpp>
#include <meltpooldg/level_set/level_set_operation.hpp>


#include <deal.II/fe/fe_system.h>

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

        adaflo = std::make_shared<AdafloWrapper<dim>>(*scratch_data, 
                                                      base_in->parameters.adaflo_params);

        // output_results(0,base_in->parameters);
        while (!time_iterator.is_finished())
        {
            const double dt = time_iterator.get_next_time_increment();
            
            adaflo->solve();
            
            convert_fe_sytem_vector_to_block_vector(adaflo->get_velocity(), dof_handler_adaflo, advection_velocity, dof_handler);
            output_results(time_iterator.get_current_time_step_number(), 
                            base_in->parameters);

            level_set_operation.solve(dt, advection_velocity);

            BlockVectorType                     surface_tension_force;
            scratch_data->initialize_dof_vector(surface_tension_force, 0);
            level_set_operation.compute_surface_tension(surface_tension_force, 
                                                        base_in->parameters.flow.surface_tension_coefficient);

            VectorType surface_out;
            scratch_data->initialize_dof_vector(surface_out, dof_adaflo_idx);
            convert_block_vector_to_fe_sytem_vector(surface_tension_force, dof_handler, surface_out, dof_handler_adaflo);
            adaflo->set_surface_tension(surface_out);
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
         *  setup DoFHandler adaflo
         */
        dof_handler_adaflo.initialize(*base_in->triangulation,
                                FESystem<dim>(FE_Q<dim>(base_in->parameters.base.degree), dim));
        /*
         *  setup DoFHandler adaflo
         */
        dof_handler.initialize(*base_in->triangulation,
                                FE_Q<dim>(base_in->parameters.base.degree));

        // scratch_data->attach_dof_handler(dof_handler_adaflo);
        scratch_data->attach_dof_handler(dof_handler);
        scratch_data->attach_dof_handler(dof_handler);
        scratch_data->attach_dof_handler(dof_handler_adaflo);
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
        for (const auto &bc : base_in->get_boundary_conditions().dirichlet_bc)
          {
            VectorTools::interpolate_boundary_values(scratch_data->get_mapping(),
                                                     dof_handler,
                                                     bc.first,
                                                     *bc.second,
                                                     constraints_dirichlet);
          }
        constraints_dirichlet.close();

        // scratch_data->attach_constraint_matrix(dummy_constraint);
        const unsigned int dof_no_bc_idx =
          scratch_data->attach_constraint_matrix(hanging_node_constraints);
        const unsigned int dof_idx        = scratch_data->attach_constraint_matrix(constraints_dirichlet);
        
        dof_adaflo_idx = scratch_data->attach_constraint_matrix(dummy_constraints);
        /*
         *  create quadrature rule
         */
        unsigned int quad_idx = scratch_data->attach_quadrature(QGauss<1>(base_in->parameters.base.n_q_points_1d));
        /*
         *  create the matrix-free object
         */
        scratch_data->build();

        time_iterator.initialize(TimeIteratorData<double>{0.0 /*start*/,
                                                    8 /*end*/,
                                                    0.02 /*dt*/,
                                                    1000 /*max_steps*/,
                                                    false /*cfl_condition-->not supported yet*/});

        scratch_data->initialize_dof_vector(advection_velocity, dof_idx);
        /*
         *  set initial conditions of the levelset function
         */
        VectorType initial_solution;
        scratch_data->initialize_dof_vector(initial_solution);
        VectorTools::project(scratch_data->get_mapping(),
                             dof_handler,
                             constraints_dirichlet,
                             scratch_data->get_quadrature(),
                             *base_in->get_field_conditions()->initial_field,
                             initial_solution);

        initial_solution.update_ghost_values();
        /*
         *    initialize the levelset operation class
         */

        
        level_set_operation.initialize( scratch_data, 
                                        initial_solution, 
                                        base_in->parameters, 
                                        dof_idx, 
                                        dof_no_bc_idx, 
                                        quad_idx);
      }

      template<int spacedim>
      static void
      convert_fe_sytem_vector_to_block_vector(const VectorType& in, const DoFHandler<dim, spacedim> & dof_handler_adaflo, BlockVectorType& out, const DoFHandler<dim, spacedim> & dof_handler)
      {
        for (const auto &cell_adaflo : dof_handler_adaflo.active_cell_iterators())
          if (cell_adaflo->is_locally_owned())
          {
              Vector<double> local(dof_handler_adaflo.get_fe().n_dofs_per_cell());
              cell_adaflo->get_dof_values(in, local);


              auto cell = DoFCellAccessor<dim, dim, false>(&dof_handler.get_triangulation(),
                                              cell_adaflo->level(), 
                                              cell_adaflo->index(),   
                                             &dof_handler);
              
              for (unsigned int d=0; d<dim; ++d)
              {
                const unsigned int n_dofs_per_component = dof_handler.get_fe().n_dofs_per_cell();
                Vector<double> local_component(n_dofs_per_component);

                for(unsigned int c = 0; c < n_dofs_per_component; ++c)
                  local_component[c] = local[c * dim + d];

                 cell.set_dof_values(local_component, out.block(d));
              }
          }

        out.update_ghost_values(); // TODO: needed?
      }

      template<int spacedim>
      static void
      convert_block_vector_to_fe_sytem_vector(const BlockVectorType& in, const DoFHandler<dim, spacedim> & dof_handler, VectorType& out, const DoFHandler<dim, spacedim> & dof_handler_adaflo)
      {
        in.update_ghost_values(); // TODO: needed?
        
        for (const auto &cell_adaflo : dof_handler_adaflo.active_cell_iterators())
          if (cell_adaflo->is_locally_owned())
          {
              auto cell = DoFCellAccessor<dim, dim, false>(&dof_handler.get_triangulation(),
                                              cell_adaflo->level(), 
                                              cell_adaflo->index(),   
                                             &dof_handler);

              Vector<double> local(dof_handler_adaflo.get_fe().n_dofs_per_cell());
              
              for (unsigned int d=0; d<dim; ++d)
              {
                 const unsigned int n_dofs_per_component = dof_handler.get_fe().n_dofs_per_cell();
                 Vector<double> local_component(n_dofs_per_component);

                 cell.get_dof_values(in.block(d), local_component);

                 for(unsigned int c = 0; c < n_dofs_per_component; ++c)
                   local[c * dim + d] = local_component[c];
              }
              cell_adaflo->set_dof_values(local, out);
          }

        out.update_ghost_values(); // TODO: needed?
      }

      /*
       *  This function is to create paraview output
       */
      void
      output_results(const unsigned int time_step, const Parameters<double> &parameters) const
      {
        // if (parameters.paraview.do_output)
        // {
        /*
         *  output advected field
        */
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);

        // data_out.add_data_vector(adaflo->get_velocity(), "velocity");
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
      }
    private:
      TimeIterator<double> time_iterator;
      DoFHandler<dim>      dof_handler_adaflo;
      DoFHandler<dim>      dof_handler;

      AffineConstraints<double> constraints_dirichlet;
      AffineConstraints<double> hanging_node_constraints;
      AffineConstraints<double> dummy_constraints;

      BlockVectorType                     advection_velocity;

      unsigned int dof_adaflo_idx;
      std::shared_ptr<ScratchData<dim>>   scratch_data;
      std::shared_ptr<AdafloWrapper<dim>> adaflo;
      LevelSet::LevelSetOperation<dim>    level_set_operation;
    };
  } // namespace TwoPhaseFlow
} // namespace MeltPoolDG