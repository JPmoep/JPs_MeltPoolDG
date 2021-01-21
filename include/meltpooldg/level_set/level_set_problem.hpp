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
#include <meltpooldg/evaporation/evaporation_operation.hpp>
#include <meltpooldg/interface/problembase.hpp>
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/level_set/level_set_operation.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/timeiterator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG
{
  namespace LevelSet
  {
    using namespace dealii;

    template <int dim>
    class LevelSetProblem : public ProblemBase<dim>
    {
    private:
      using VectorType      = LinearAlgebra::distributed::Vector<double>;
      using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    public:
      LevelSetProblem() = default;

      void
      run(std::shared_ptr<SimulationBase<dim>> base_in) final
      {
        initialize(base_in);

        while (!time_iterator.is_finished())
          {
            const double dt = time_iterator.get_next_time_increment();
            scratch_data->get_pcout()
              << "| ls: t= " << std::setw(10) << std::left << time_iterator.get_current_time();
            compute_advection_velocity(*base_in->get_advection_field("level_set"));

            if (base_in->parameters.base.problem_name == "level_set_with_evaporation")
              {
                /**
                 * If evaporative mass flux is considered the interface velocity will be modified.
                 * Note that the normal vector is used from the old step.
                 */
                level_set_operation.update_normal_vector();
                evaporation_operation->solve();
                advection_velocity += evaporation_operation->get_evaporation_velocity();
              }
            level_set_operation.solve(dt, advection_velocity);

            // do paraview output if requested
            output_results(time_iterator.get_current_time_step_number(),
                           base_in->parameters.base.problem_name == "level_set_with_evaporation");

            if (base_in->parameters.amr.do_amr)
              refine_mesh(base_in);
          }
      }

      std::string
      get_name() final
      {
        return "level_set_problem";
      }

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
        scratch_data = std::make_shared<ScratchData<dim>>(base_in->parameters.ls.do_matrix_free);
        /*
         *  setup mapping
         */
#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
        if (base_in->parameters.base.do_simplex)
          scratch_data->set_mapping(
            MappingFE<dim>(Simplex::FE_P<dim>(base_in->parameters.base.degree)));
        else
#endif
          scratch_data->set_mapping(MappingQGeneric<dim>(base_in->parameters.base.degree));
        /*
         *  setup DoFHandler
         */
        dof_handler.reinit(*base_in->triangulation);
        dof_handler_velocity.reinit(*base_in->triangulation);

        this->ls_hanging_nodes_dof_idx = scratch_data->attach_dof_handler(dof_handler);
        this->ls_dof_idx               = scratch_data->attach_dof_handler(dof_handler);
        ls_zero_bc_idx                 = scratch_data->attach_dof_handler(dof_handler);
        vel_dof_idx                    = scratch_data->attach_dof_handler(dof_handler_velocity);

        scratch_data->attach_constraint_matrix(hanging_node_constraints);
        scratch_data->attach_constraint_matrix(constraints_dirichlet);
        scratch_data->attach_constraint_matrix(hanging_node_constraints_with_zero_dirichlet);
        scratch_data->attach_constraint_matrix(hanging_node_constraints_velocity);

        /*
         *  create quadrature rule
         */

#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
        if (base_in->parameters.base.do_simplex)
          ls_quad_idx = scratch_data->attach_quadrature(
            Simplex::QGauss<dim>(base_in->parameters.base.n_q_points_1d));
        else
#endif
          ls_quad_idx =
            scratch_data->attach_quadrature(QGauss<1>(base_in->parameters.base.n_q_points_1d));

          // TODO: only do once!
#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
        if (base_in->parameters.base.do_simplex)
          scratch_data->attach_quadrature(
            Simplex::QGauss<dim>(base_in->parameters.base.n_q_points_1d));
        else
#endif
          scratch_data->attach_quadrature(QGauss<1>(base_in->parameters.base.n_q_points_1d));
        /*
         *  initialize the time iterator
         */
        time_iterator.initialize(TimeIteratorData<double>{base_in->parameters.ls.start_time,
                                                          base_in->parameters.ls.end_time,
                                                          base_in->parameters.ls.time_step_size,
                                                          100000,
                                                          false});

        setup_dof_system(base_in, false);

        // set initial conditions of the levelset function
        AssertThrow(
          base_in->get_initial_condition("level_set"),
          ExcMessage(
            "It seems that your SimulationBase object does not contain "
            "a valid initial field function for the level set field. A shared_ptr to your initial field "
            "function, e.g., MyInitializeFunc<dim> must be specified as follows: "
            "this->attach_initial_condition(std::make_shared<MyInitializeFunc<dim>>(), "
            "'level_set') "));

        scratch_data->initialize_dof_vector(initial_solution);
        dealii::VectorTools::project(scratch_data->get_mapping(),
                                     dof_handler,
                                     constraints_dirichlet,
                                     scratch_data->get_quadrature(),
                                     *base_in->get_initial_condition("level_set"),
                                     initial_solution);

        initial_solution.update_ghost_values();

        level_set_operation.initialize(scratch_data,
                                       initial_solution,
                                       advection_velocity,
                                       base_in,
                                       ls_dof_idx,
                                       ls_hanging_nodes_dof_idx,
                                       ls_quad_idx,
                                       reinit_dof_idx,
                                       reinit_dof_idx,
                                       curv_dof_idx,
                                       normal_dof_idx,
                                       vel_dof_idx,
                                       ls_zero_bc_idx);

        if (base_in->parameters.base.problem_name == "level_set_with_evaporation")
          {
            evaporation_operation = std::make_shared<Evaporation::EvaporationOperation<dim>>(
              scratch_data,
              advection_velocity,
              level_set_operation.get_level_set(),
              level_set_operation.get_normal_vector(),
              base_in,
              normal_dof_idx,
              vel_dof_idx,
              ls_quad_idx,
              ls_dof_idx);
          }
        /*
         *  initialize postprocessor
         */
        post_processor =
          std::make_shared<Postprocessor<dim>>(scratch_data->get_mpi_comm(ls_dof_idx),
                                               base_in->parameters.paraview,
                                               scratch_data->get_mapping(),
                                               scratch_data->get_triangulation(ls_dof_idx));

        // initialize variables
        output_results(0, base_in->parameters.base.problem_name == "melt_pool");
        /*
         *    Do initial refinement steps if requested
         */
        if (base_in->parameters.amr.do_amr &&
            base_in->parameters.amr.n_initial_refinement_cycles > 0)
          for (int i = 0; i < base_in->parameters.amr.n_initial_refinement_cycles; ++i)
            {
              scratch_data->get_pcout()
                << "cycle: " << i << " n_dofs: " << dof_handler.n_dofs() << "(ls)" << std::endl;
              refine_mesh(base_in);
            }
      }

      void
      setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in, const bool do_reinit = true)
      {
#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
        if (base_in->parameters.base.do_simplex)
          {
            dof_handler.distribute_dofs(Simplex::FE_P<dim>(base_in->parameters.base.degree));
            dof_handler_velocity.distribute_dofs(
              FESystem<dim>(Simplex::FE_P<dim>(base_in->parameters.base.degree), dim));
          }
        else
#endif
          {
            dof_handler.distribute_dofs(FE_Q<dim>(base_in->parameters.base.degree));
            dof_handler_velocity.distribute_dofs(
              FESystem<dim>(FE_Q<dim>(base_in->parameters.base.degree), dim));
          }
        /*
         *  create partitioning
         */
        scratch_data->create_partitioning();
        /*
         *  make hanging nodes constraints
         */

        /*
         *  make hanging nodes and dirichlet constraints (at the moment no time-dependent
         *  dirichlet constraints are supported)
         */
        hanging_node_constraints.clear();
        hanging_node_constraints.reinit(
          scratch_data->get_locally_relevant_dofs(ls_hanging_nodes_dof_idx));
        DoFTools::make_hanging_node_constraints(dof_handler, hanging_node_constraints);
        hanging_node_constraints.close();

        hanging_node_constraints_velocity.clear();
        hanging_node_constraints_velocity.reinit(
          scratch_data->get_locally_relevant_dofs(vel_dof_idx));
        DoFTools::make_hanging_node_constraints(dof_handler_velocity,
                                                hanging_node_constraints_velocity);
        hanging_node_constraints_velocity.close();

        constraints_dirichlet.clear();
        constraints_dirichlet.reinit(scratch_data->get_locally_relevant_dofs(ls_dof_idx));
        constraints_dirichlet.merge(
          hanging_node_constraints,
          AffineConstraints<double>::MergeConflictBehavior::left_object_wins);
        for (const auto &bc : base_in->get_dirichlet_bc(
               "level_set")) // @todo: add name of bc at a more central place
          {
            dealii::VectorTools::interpolate_boundary_values(scratch_data->get_mapping(),
                                                             dof_handler,
                                                             bc.first,
                                                             *bc.second,
                                                             constraints_dirichlet);
          }
        constraints_dirichlet.close();

        hanging_node_constraints_with_zero_dirichlet.clear();
        hanging_node_constraints_with_zero_dirichlet.reinit(
          scratch_data->get_locally_relevant_dofs(ls_zero_bc_idx));
        DoFTools::make_hanging_node_constraints(dof_handler,
                                                hanging_node_constraints_with_zero_dirichlet);
        for (const auto &bc : base_in->get_dirichlet_bc(
               "level_set")) // @todo: add name of bc at a more central place
          {
            dealii::DoFTools::make_zero_boundary_constraints(
              dof_handler, bc.first, hanging_node_constraints_with_zero_dirichlet);
          }
        hanging_node_constraints_with_zero_dirichlet.close();
        /*
         *  create the matrix-free object
         */
        scratch_data->build();

        // initialize the levelset operation class
        AssertThrow(base_in->get_advection_field("level_set"),
                    ExcMessage(
                      " It seems that your SimulationBase object does not contain "
                      "a valid advection velocity. A shared_ptr to your advection velocity "
                      "function, e.g., AdvectionFunc<dim> must be specified as follows: "
                      "this->attach_advection_field(std::make_shared<AdvecFunc<dim>>(), "
                      "'level_set') "));
        compute_advection_velocity(*base_in->get_advection_field("level_set"));

        if (do_reinit)
          {
            level_set_operation.reinit();
            evaporation_operation->reinit();
          }
      }

      void
      compute_advection_velocity(Function<dim> &advec_func)
      {
        scratch_data->initialize_dof_vector(advection_velocity, vel_dof_idx);
        /*
         *  set the current time to the advection field function
         */
        advec_func.set_time(time_iterator.get_current_time());

        dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                         scratch_data->get_dof_handler(vel_dof_idx),
                                         advec_func,
                                         advection_velocity);
        advection_velocity.update_ghost_values();
      }
      /*
       *  This function is to create paraview output
       */
      void
      output_results(const unsigned int time_step, bool do_evaporation) const
      {
        const auto attach_output_vectors = [&](DataOut<dim> &data_out) {
          level_set_operation.attach_output_vectors(data_out);
          if (do_evaporation)
            evaporation_operation->attach_output_vectors(data_out);
          /*
           *  output advection velocity
           */
          MeltPoolDG::VectorTools::update_ghost_values(advection_velocity);
          std::vector<DataComponentInterpretation::DataComponentInterpretation>
            vector_component_interpretation(
              dim, DataComponentInterpretation::component_is_part_of_vector);

          data_out.add_data_vector(dof_handler_velocity,
                                   advection_velocity,
                                   std::vector<std::string>(dim, "velocity"),
                                   vector_component_interpretation);
        };
        post_processor->process(time_step, attach_output_vectors);
      }

      /*
       *  perform mesh refinement
       */
      void
      refine_mesh(std::shared_ptr<SimulationBase<dim>> base_in)
      {
        const auto mark_cells_for_refinement =
          [&](parallel::distributed::Triangulation<dim> &tria) -> bool {
          Vector<float> estimated_error_per_cell(base_in->triangulation->n_active_cells());

          VectorType locally_relevant_solution;
          locally_relevant_solution.reinit(scratch_data->get_partitioner(ls_dof_idx));
          locally_relevant_solution.copy_locally_owned_data_from(
            level_set_operation.get_level_set());
          constraints_dirichlet.distribute(locally_relevant_solution);
          locally_relevant_solution.update_ghost_values();

          for (unsigned int i = 0; i < locally_relevant_solution.local_size(); ++i)
            locally_relevant_solution.local_element(i) =
              (1.0 - locally_relevant_solution.local_element(i) *
                       locally_relevant_solution.local_element(i));

          locally_relevant_solution.update_ghost_values();

          dealii::VectorTools::integrate_difference(scratch_data->get_dof_handler(ls_dof_idx),
                                                    locally_relevant_solution,
                                                    Functions::ZeroFunction<dim>(),
                                                    estimated_error_per_cell,
                                                    scratch_data->get_quadrature(ls_quad_idx),
                                                    dealii::VectorTools::L2_norm);

          parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
            tria,
            estimated_error_per_cell,
            base_in->parameters.amr.upper_perc_to_refine,
            base_in->parameters.amr.lower_perc_to_coarsen);

          return true;
        };

        const auto attach_vectors = [&](std::vector<VectorType *> &vectors) {
          level_set_operation.attach_vectors(vectors);
        };

        const auto post = [&]() {
          constraints_dirichlet.distribute(level_set_operation.get_level_set());
        };

        const auto setup_dof_system = [&]() { this->setup_dof_system(base_in); };

        refine_grid<dim, VectorType>(mark_cells_for_refinement,
                                     attach_vectors,
                                     post,
                                     setup_dof_system,
                                     base_in->parameters.amr,
                                     dof_handler,
                                     time_iterator.get_current_time_step_number());
      }

    private:
      DoFHandler<dim>           dof_handler;
      DoFHandler<dim>           dof_handler_velocity;
      AffineConstraints<double> constraints_dirichlet;
      AffineConstraints<double> hanging_node_constraints;
      AffineConstraints<double> hanging_node_constraints_velocity;
      AffineConstraints<double> hanging_node_constraints_with_zero_dirichlet;

      std::shared_ptr<ScratchData<dim>> scratch_data;
      VectorType                        advection_velocity;

      TimeIterator<double>                                    time_iterator;
      LevelSetOperation<dim>                                  level_set_operation;
      std::shared_ptr<Evaporation::EvaporationOperation<dim>> evaporation_operation;
      VectorType                                              initial_solution;
      unsigned int                                            ls_dof_idx;
      unsigned int                                            ls_quad_idx;
      unsigned int                                            ls_zero_bc_idx;
      unsigned int                                            ls_hanging_nodes_dof_idx;
      unsigned int                                            vel_dof_idx;
      const unsigned int &curv_dof_idx   = ls_hanging_nodes_dof_idx;
      const unsigned int &normal_dof_idx = ls_hanging_nodes_dof_idx;
      const unsigned int &reinit_dof_idx =
        ls_hanging_nodes_dof_idx; //@todo: would it make sense to use ls_zero_bc_idx?
      const unsigned int &reinit_hanging_nodes_dof_idx =
        ls_hanging_nodes_dof_idx; //@todo: would it make sense to use ls_zero_bc_idx?

      std::shared_ptr<Postprocessor<dim>> post_processor;
    };
  } // namespace LevelSet
} // namespace MeltPoolDG
