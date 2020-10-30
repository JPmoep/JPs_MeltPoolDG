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

        // TODO: make class field?
        BlockVectorType surface_tension_force;
        scratch_data->initialize_dof_vector(surface_tension_force, dof_idx);

        // TODO: re-enable?
        // output_results(0,base_in->parameters);
        while (!time_iterator.is_finished())
          {
            const auto dt = time_iterator.get_next_time_increment();
            const auto n  = time_iterator.get_current_time_step_number();

            flow_operation->solve();

            flow_operation->get_velocity(advection_velocity);

            // TODO: why here?
            output_results(n, base_in->parameters);

            level_set_operation.solve(dt, advection_velocity);
            level_set_operation.compute_surface_tension(
              surface_tension_force, base_in->parameters.flow.surface_tension_coefficient);


            flow_operation->set_surface_tension(surface_tension_force);
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
        for (const auto &bc : base_in->get_boundary_conditions().dirichlet_bc)
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
          TimeIteratorData<double>{0.0 /*start*/,
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
                                                              base_in->parameters.adaflo_params);
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
