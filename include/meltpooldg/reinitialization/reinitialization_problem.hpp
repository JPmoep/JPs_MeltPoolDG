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
// for distributed triangulation
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria_base.h>
// for dof_handler type
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/numerics/error_estimator.h>
// for FE_Q<dim> type
#include <deal.II/fe/fe_q.h>
// for mapping
#include <deal.II/fe/mapping.h>
// for simplex
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/simplex/fe_lib.h>
#include <deal.II/simplex/quadrature_lib.h>
// MeltPoolDG
#include <meltpooldg/interface/problembase.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/reinitialization/reinitialization_operation.hpp>
#include <meltpooldg/utilities/timeiterator.hpp>
// C++
#include <memory>
namespace MeltPoolDG
{
  namespace Reinitialization
  {
    using namespace dealii;

    /*
     *     Reinitialization model for reobtaining the "signed-distance"
     *     property of the level set equation
     */

    template <int dim>
    class ReinitializationProblem : public ProblemBase<dim>
    {
    private:
      using VectorType     = LinearAlgebra::distributed::Vector<double>;
      using DoFHandlerType = DoFHandler<dim>;

    public:
      /*
       *  Constructor of reinitialization problem
       */

      ReinitializationProblem() = default;

      void
      run(std::shared_ptr<SimulationBase<dim>> base_in) final
      {
        initialize(base_in);

        while (!time_iterator.is_finished())
          {
            const double d_tau = time_iterator.get_next_time_increment();
            scratch_data->get_pcout()
              << "t= " << std::setw(10) << std::left << time_iterator.get_current_time();

            reinit_operation.solve(d_tau);

            output_results(time_iterator.get_current_time_step_number(), base_in->parameters);

            if (base_in->parameters.amr.do_amr)
              refine_mesh(base_in);
          }
      }

      std::string
      get_name() final
      {
        return "reinitialization";
      };

    private:
      /*
       *  This function initials the relevant member data
       *  for the computation of a reinitialization problem
       */
      void
      initialize(std::shared_ptr<SimulationBase<dim>> base_in)
      {
        /*
         *  initialize the dof system
         */
        setup_dof_system(base_in, true);
        /*
         *  initialize the time iterator
         */
        time_iterator.initialize(TimeIteratorData<double>{0.0,
                                                          10000.,
                                                          base_in->parameters.reinit.dtau,
                                                          base_in->parameters.reinit.max_n_steps,
                                                          false});
        /*
         *  set initial conditions of the levelset function
         */
        VectorType solution_level_set;
        scratch_data->initialize_dof_vector(solution_level_set);
        VectorTools::project(scratch_data->get_mapping(),
                             dof_handler,
                             constraints,
                             scratch_data->get_quadrature(),
                             *base_in->get_field_conditions()->initial_field,
                             solution_level_set);

        solution_level_set.update_ghost_values();

        /*
         *    initialize the reinitialization operation class
         */
        reinit_operation.initialize(
          scratch_data, solution_level_set, base_in->parameters, dof_idx, quad_idx);
      }

      void
      setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in, const bool do_initial_setup)
      {
        /*
         *  setup scratch data
         */
        if (do_initial_setup)
          {
            scratch_data =
              std::make_shared<ScratchData<dim>>(base_in->parameters.reinit.do_matrix_free);
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
              scratch_data->attach_quadrature(Simplex::QGauss<dim>(
                dim == 2 ? (base_in->parameters.base.n_q_points_1d == 1 ? 3 : 7) :
                           (base_in->parameters.base.n_q_points_1d == 1 ? 4 : 10)));
            else
#endif
              quad_idx =
                scratch_data->attach_quadrature(QGauss<1>(base_in->parameters.base.n_q_points_1d));
          }
          /*
           *  setup DoFHandler
           */
#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
        if (base_in->parameters.base.do_simplex)
          dof_handler.initialize(*base_in->triangulation,
                                 Simplex::FE_P<dim>(base_in->parameters.base.degree));
        else
#endif
          dof_handler.initialize(*base_in->triangulation,
                                 FE_Q<dim>(base_in->parameters.base.degree));

        if (do_initial_setup)
          scratch_data->attach_dof_handler(dof_handler);
        /*
         *  re-create partitioning
         */
        scratch_data->create_partitioning();
        /*
         *  make hanging nodes constraints
         */
        constraints.clear();
        constraints.reinit(scratch_data->get_locally_relevant_dofs());
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        constraints.close();

        if (do_initial_setup)
          dof_idx = scratch_data->attach_constraint_matrix(constraints);

        /*
         *  create the matrix-free object
         */
        scratch_data->build();
      }

      /*
       *  perform mesh refinement
       */
      void
      refine_mesh(std::shared_ptr<SimulationBase<dim>> base_in)
      {
        if (auto tria = std::dynamic_pointer_cast<parallel::distributed::Triangulation<dim>>(
              base_in->triangulation))
          {
            Vector<float> estimated_error_per_cell(base_in->triangulation->n_active_cells());

            /*  @todo:
             *  bug (?)
             *  for the purpose of the KellyErrorEstimator initialize_dof_vector could not be used
             *  scratch_data->initialize_dof_vector(locally_relevant_solution);
             */

            VectorType locally_relevant_solution;
            locally_relevant_solution.reinit(scratch_data->get_partitioner());
            locally_relevant_solution.copy_locally_owned_data_from(
              reinit_operation.solution_level_set);
            constraints.distribute(locally_relevant_solution);
            locally_relevant_solution.update_ghost_values();

            KellyErrorEstimator<dim>::estimate(scratch_data->get_dof_handler(),
                                               scratch_data->get_face_quadrature(),
                                               {},
                                               locally_relevant_solution,
                                               estimated_error_per_cell);

            parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
              *tria,
              estimated_error_per_cell,
              base_in->parameters.amr.upper_perc_to_refine,
              base_in->parameters.amr.lower_perc_to_coarsen);

            /*
             *  Limit the maximum and minimum refinement levels of cells of the grid.
             */
            if (tria->n_levels() > base_in->parameters.amr.max_grid_refinement_level)
              for (auto &cell : tria->active_cell_iterators_on_level(
                     base_in->parameters.amr.max_grid_refinement_level))
                cell->clear_refine_flag();
            if (tria->n_levels() < base_in->parameters.amr.min_grid_refinement_level)
              for (auto &cell : tria->active_cell_iterators_on_level(
                     base_in->parameters.amr.min_grid_refinement_level))
                cell->clear_coarsen_flag();

            /*
             *  Initialize the triangulation change from the old grid to the new grid
             */
            base_in->triangulation->prepare_coarsening_and_refinement();
            /*
             *  Initialize the solution transfer from the old grid to the new grid
             */
            parallel::distributed::SolutionTransfer<dim, VectorType, DoFHandlerType>
              solution_transfer(dof_handler);
            solution_transfer.prepare_for_coarsening_and_refinement(locally_relevant_solution);

            /*
             *  Execute the grid refinement
             */
            base_in->triangulation->execute_coarsening_and_refinement();

            /*
             *  update dof-related scratch data to match the current triangulation
             */
            setup_dof_system(base_in, false);

            /*
             *  interpolate the given solution to the new discretization
             *
             */
            VectorType interpolated_solution;
            scratch_data->initialize_dof_vector(interpolated_solution);

            solution_transfer.interpolate(interpolated_solution);

            constraints.distribute(interpolated_solution);
            /*
             * update the reinitialization operator with the new solution values
             */
            reinit_operation.update_initial_solution(interpolated_solution);
          }
        else
          //@todo: WIP
          AssertThrow(false, ExcMessage("Mesh refinement for dim=1 not yet supported"));
      }

      /*
       *  Creating paraview output
       */
      void
      output_results(const unsigned int time_step, const Parameters<double> &parameters) const
      {
        if (parameters.paraview.do_output)
          {
            DataOut<dim> data_out;
            data_out.attach_dof_handler(dof_handler);
            data_out.add_data_vector(reinit_operation.solution_level_set, "psi");

            if (parameters.paraview.print_normal_vector)
              {
                for (unsigned int d = 0; d < dim; ++d)
                  data_out.add_data_vector(reinit_operation.solution_normal_vector.block(d),
                                           "normal_" + std::to_string(d));
              }

            data_out.build_patches(scratch_data->get_mapping());
            data_out.write_vtu_with_pvtu_record("./",
                                                parameters.paraview.filename,
                                                time_step,
                                                scratch_data->get_mpi_comm(),
                                                parameters.paraview.n_digits_timestep,
                                                parameters.paraview.n_groups);
          }
      }

    private:
      DoFHandler<dim>           dof_handler;
      AffineConstraints<double> constraints;

      std::shared_ptr<ScratchData<dim>> scratch_data;
      TimeIterator<double>              time_iterator;
      ReinitializationOperation<dim>    reinit_operation;
      unsigned int                      dof_idx;
      unsigned int                      quad_idx;
    };
  } // namespace Reinitialization
} // namespace MeltPoolDG
