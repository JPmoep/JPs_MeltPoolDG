/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/index_set.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/simplex/fe_lib.h>
#include <deal.II/simplex/quadrature_lib.h>

// MeltPoolDG
#include <meltpooldg/advection_diffusion/advection_diffusion_adaflo_wrapper.hpp>
#include <meltpooldg/advection_diffusion/advection_diffusion_operation.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/interface/problembase.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/postprocessor.hpp>
#include <meltpooldg/utilities/timeiterator.hpp>

namespace MeltPoolDG
{
  namespace AdvectionDiffusion
  {
    using namespace dealii;

    template <int dim>
    class AdvectionDiffusionProblem : public ProblemBase<dim>
    {
    private:
      using VectorType      = LinearAlgebra::distributed::Vector<double>;
      using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    public:
      AdvectionDiffusionProblem() = default;

      void
      run(std::shared_ptr<SimulationBase<dim>> base_in) final
      {
        AssertThrow(base_in->get_advection_field("advection_diffusion"),
                    ExcMessage(
                      " It seems that your SimulationBase object does not contain "
                      "a valid advection velocity. A shared_ptr to your advection velocity "
                      "function, e.g., AdvectionFunc<dim> must be specified as follows: "
                      "this->attach_advection_field(std::make_shared<AdvecFunc<dim>>(), "
                      "'advection_diffusion') "));

        initialize(base_in);

        while (!time_iterator.is_finished())
          {
            const double dt = time_iterator.get_next_time_increment();
            scratch_data->get_pcout()
              << "t= " << std::setw(10) << std::left << time_iterator.get_current_time();
            /*
             * compute the advection velocity for the current time
             */
            compute_advection_velocity(*base_in->get_advection_field("advection_diffusion"));
            advec_diff_operation->solve(dt, advection_velocity);
            /*
             *  do paraview output if requested
             */
            output_results(time_iterator.get_current_time_step_number());

            if (base_in->parameters.amr.do_amr)
              refine_mesh(base_in);
          }
      }

      std::string
      get_name() final
      {
        return "advection-diffusion problem";
      };

    private:
      /*
       *  This function initials the relevant member data
       *  for the computation of the advection-diffusion problem
       */
      void
      setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in)
      {
        /*
         *  setup DoFHandler
         */
        dof_handler.distribute_dofs(*fe);
        dof_handler_velocity.distribute_dofs(*fe_velocity);

        /*
         *  create the partititioning
         */
        scratch_data->create_partitioning();
        /*
         *  make hanging nodes and dirichlet constraints (Note: at the moment no time-dependent
         *  dirichlet constraints are supported)
         */
        hanging_node_constraints.clear();
        hanging_node_constraints.reinit(
          scratch_data->get_locally_relevant_dofs(advec_diff_dof_idx));
        DoFTools::make_hanging_node_constraints(dof_handler, hanging_node_constraints);
        hanging_node_constraints.close();

        hanging_node_constraints_with_zero_dirichlet.clear();
        hanging_node_constraints_with_zero_dirichlet.reinit(
          scratch_data->get_locally_relevant_dofs(advec_diff_adaflo_dof_idx));
        DoFTools::make_hanging_node_constraints(dof_handler,
                                                hanging_node_constraints_with_zero_dirichlet);
        for (const auto &bc : base_in->get_dirichlet_bc(
               "advection_diffusion")) // @todo: add name of bc at a more central place
          {
            dealii::DoFTools::make_zero_boundary_constraints(
              dof_handler, bc.first, hanging_node_constraints_with_zero_dirichlet);
          }
        hanging_node_constraints_with_zero_dirichlet.close();

        hanging_node_constraints_velocity.clear();
        hanging_node_constraints_velocity.reinit(
          scratch_data->get_locally_relevant_dofs(velocity_dof_idx));
        DoFTools::make_hanging_node_constraints(dof_handler_velocity,
                                                hanging_node_constraints_velocity);
        hanging_node_constraints_velocity.close();

        constraints.clear();
        constraints.reinit(scratch_data->get_locally_relevant_dofs(advec_diff_dof_idx));
        constraints.merge(hanging_node_constraints,
                          AffineConstraints<double>::MergeConflictBehavior::left_object_wins);
        for (const auto &bc : base_in->get_dirichlet_bc(
               "advection_diffusion")) // @todo: add name of bc at a more central place
          {
            dealii::VectorTools::interpolate_boundary_values(
              scratch_data->get_mapping(), dof_handler, bc.first, *bc.second, constraints);
          }
        constraints.close();

        /*
         *  create the matrix-free object
         */
        scratch_data->build();

        if (advec_diff_operation) // TODO: better place
          advec_diff_operation->reinit();
      }


      void
      initialize(std::shared_ptr<SimulationBase<dim>> base_in)
      {
        /*
         *  setup DoFHandler
         */
        dof_handler.reinit(*base_in->triangulation);
        dof_handler_velocity.reinit(*base_in->triangulation);

#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
        if (base_in->parameters.base.do_simplex)
          {
            fe = std::make_unique<Simplex::FE_P<dim>>(base_in->parameters.base.degree);
            fe_velocity =
              std::make_unique<FESystem<dim>>(Simplex::FE_P<dim>(base_in->parameters.base.degree),
                                              dim);
          }
        else
#endif
          {
            fe = std::make_unique<FE_Q<dim>>(base_in->parameters.base.degree);
            fe_velocity =
              std::make_unique<FESystem<dim>>(FE_Q<dim>(base_in->parameters.base.degree), dim);
          }

        /*
         *  setup scratch data
         */
        {
          scratch_data =
            std::make_shared<ScratchData<dim>>(base_in->parameters.advec_diff.do_matrix_free);
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
             *  create quadrature rule
             */
#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
          if (base_in->parameters.base.do_simplex)
            advec_diff_quad_idx = scratch_data->attach_quadrature(
              Simplex::QGauss<dim>(base_in->parameters.base.n_q_points_1d));
          else
#endif
            advec_diff_quad_idx =
              scratch_data->attach_quadrature(QGauss<1>(base_in->parameters.base.n_q_points_1d));

          advec_diff_dof_idx               = scratch_data->attach_dof_handler(dof_handler);
          advec_diff_hanging_nodes_dof_idx = scratch_data->attach_dof_handler(dof_handler);
          advec_diff_adaflo_dof_idx        = scratch_data->attach_dof_handler(dof_handler);
          velocity_dof_idx                 = scratch_data->attach_dof_handler(dof_handler_velocity);

          scratch_data->attach_constraint_matrix(constraints);
          scratch_data->attach_constraint_matrix(hanging_node_constraints);
          scratch_data->attach_constraint_matrix(hanging_node_constraints_with_zero_dirichlet);
          scratch_data->attach_constraint_matrix(hanging_node_constraints_velocity);
        }

        setup_dof_system(base_in);

        /*
         *  initialize the time iterator
         */
        time_iterator.initialize(
          TimeIteratorData<double>{base_in->parameters.advec_diff.start_time,
                                   base_in->parameters.advec_diff.end_time,
                                   base_in->parameters.advec_diff.time_step_size,
                                   base_in->parameters.advec_diff.max_n_steps,
                                   false});

        /*
         *  set initial conditions of the levelset function
         */
        AssertThrow(
          base_in->get_initial_condition("advection_diffusion"),
          ExcMessage(
            "It seems that your SimulationBase object does not contain "
            "a valid initial field function for the level set field. A shared_ptr to your initial field "
            "function, e.g., MyInitializeFunc<dim> must be specified as follows: "
            "this->attach_initial_condition(std::make_shared<MyInitializeFunc<dim>>(), "
            "'advection_diffusion') "));
        VectorType initial_solution;
        scratch_data->initialize_dof_vector(initial_solution);

        dealii::VectorTools::project(scratch_data->get_mapping(),
                                     dof_handler,
                                     constraints,
                                     scratch_data->get_quadrature(),
                                     *base_in->get_initial_condition("advection_diffusion"),
                                     initial_solution);

        initial_solution.update_ghost_values();
        /*
         *    initialize the advection-diffusion operation class
         */
        compute_advection_velocity(*base_in->get_advection_field("advection_diffusion"));
        if (base_in->parameters.advec_diff.implementation == "meltpooldg")
          {
            advec_diff_operation = std::make_shared<AdvectionDiffusionOperation<dim>>();

            advec_diff_operation->initialize(scratch_data,
                                             initial_solution,
                                             base_in->parameters,
                                             advec_diff_dof_idx,
                                             advec_diff_hanging_nodes_dof_idx,
                                             advec_diff_quad_idx,
                                             velocity_dof_idx);
          }
#ifdef MELT_POOL_DG_WITH_ADAFLO
        else if (base_in->parameters.advec_diff.implementation == "adaflo")
          {
            AssertThrow(base_in->parameters.advec_diff.do_matrix_free, ExcNotImplemented());
            advec_diff_operation =
              std::make_shared<AdvectionDiffusionOperationAdaflo<dim>>(*scratch_data,
                                                                       advec_diff_adaflo_dof_idx,
                                                                       advec_diff_quad_idx,
                                                                       velocity_dof_idx,
                                                                       base_in);
            advec_diff_operation->reinit();

            advec_diff_operation->set_initial_condition(initial_solution, advection_velocity);
          }
#endif
        else
          AssertThrow(false, ExcNotImplemented());
        /*
         *  initialize postprocessor
         */
        post_processor =
          std::make_shared<Postprocessor<dim>>(scratch_data->get_mpi_comm(advec_diff_dof_idx),
                                               base_in->parameters.paraview,
                                               scratch_data->get_mapping(),
                                               scratch_data->get_triangulation(advec_diff_dof_idx));
      }

      void
      compute_advection_velocity(Function<dim> &advec_func)
      {
        scratch_data->initialize_dof_vector(advection_velocity, velocity_dof_idx);
        /*
         *  set the current time to the advection field function
         */
        advec_func.set_time(time_iterator.get_current_time());
        /*
         *  interpolate the values of the advection velocity
         */
        dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                         scratch_data->get_dof_handler(velocity_dof_idx),
                                         advec_func,
                                         advection_velocity);
      }

      /*
       *  perform output of results
       */
      void
      output_results(const unsigned int time_step)
      {
        const auto attach_output_vectors = [&](DataOut<dim> &data_out) {
          advec_diff_operation->attach_output_vectors(data_out);

          std::vector<DataComponentInterpretation::DataComponentInterpretation>
            vector_component_interpretation(
              dim, DataComponentInterpretation::component_is_part_of_vector);
          advection_velocity.update_ghost_values();
          data_out.add_data_vector(scratch_data->get_dof_handler(velocity_dof_idx),
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
          locally_relevant_solution.reinit(scratch_data->get_partitioner(advec_diff_dof_idx));
          locally_relevant_solution.copy_locally_owned_data_from(
            advec_diff_operation->get_advected_field());
          constraints.distribute(locally_relevant_solution);
          locally_relevant_solution.update_ghost_values();

          KellyErrorEstimator<dim>::estimate(scratch_data->get_dof_handler(advec_diff_dof_idx),
                                             scratch_data->get_face_quadrature(advec_diff_quad_idx),
                                             {},
                                             locally_relevant_solution,
                                             estimated_error_per_cell);

          parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
            tria,
            estimated_error_per_cell,
            base_in->parameters.amr.upper_perc_to_refine,
            base_in->parameters.amr.lower_perc_to_coarsen);

          return true;
        };

        const auto attach_vectors = [&](std::vector<VectorType *> &vectors) {
          advec_diff_operation->attach_vectors(vectors);
        };

        const auto post = [&]() {
          constraints.distribute(advec_diff_operation->get_advected_field());
        };

        const auto setup_dof_system = [&]() { this->setup_dof_system(base_in); };

        refine_grid<dim, VectorType>(mark_cells_for_refinement,
                                     attach_vectors,
                                     post,
                                     setup_dof_system,
                                     base_in->parameters.amr,
                                     dof_handler);
        constraints.distribute(advec_diff_operation->get_advected_field());
      }

    private:
      std::unique_ptr<FiniteElement<dim>> fe;
      std::unique_ptr<FiniteElement<dim>> fe_velocity;

      DoFHandler<dim>                   dof_handler;
      AffineConstraints<double>         constraints;
      AffineConstraints<double>         hanging_node_constraints;
      AffineConstraints<double>         hanging_node_constraints_with_zero_dirichlet;
      DoFHandler<dim>                   dof_handler_velocity;
      AffineConstraints<double>         hanging_node_constraints_velocity;
      std::shared_ptr<ScratchData<dim>> scratch_data;
      VectorType                        advection_velocity;
      TimeIterator<double>              time_iterator;
      std::shared_ptr<AdvectionDiffusionOperationBase<dim>> advec_diff_operation;

      unsigned int advec_diff_dof_idx;
      unsigned int advec_diff_hanging_nodes_dof_idx;
      unsigned int advec_diff_adaflo_dof_idx;
      unsigned int velocity_dof_idx;

      unsigned int advec_diff_quad_idx;

      std::shared_ptr<Postprocessor<dim>> post_processor;
    };
  } // namespace AdvectionDiffusion
} // namespace MeltPoolDG
