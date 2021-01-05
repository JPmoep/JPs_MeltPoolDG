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
#include <meltpooldg/reinitialization/reinitialization_operation_adaflo_wrapper.hpp>
#include <meltpooldg/reinitialization/reinitialization_operation_base.hpp>
#include <meltpooldg/utilities/amr.hpp>
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

            reinit_operation->solve(d_tau);

            output_results(time_iterator.get_current_time_step_number());

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
         *  setup scratch data
         */
        {
          scratch_data =
            std::make_shared<ScratchData<dim>>(base_in->parameters.reinit.solver.do_matrix_free);
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
            scratch_data->attach_quadrature(
              Simplex::QGauss<dim>(base_in->parameters.base.n_q_points_1d));
          else
#endif
            reinit_quad_idx =
              scratch_data->attach_quadrature(QGauss<1>(base_in->parameters.base.n_q_points_1d));

          scratch_data->attach_dof_handler(dof_handler);
          reinit_dof_idx = scratch_data->attach_constraint_matrix(constraints);
          normal_dof_idx = reinit_dof_idx;
          /*
           *  setup DoFHandler
           */
          dof_handler.reinit(*base_in->triangulation);
        }

        setup_dof_system(base_in);

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
        AssertThrow(
          base_in->get_initial_condition("level_set"),
          ExcMessage(
            "It seems that your SimulationBase object does not contain "
            "a valid initial field function for the level set field. A shared_ptr to your initial field "
            "function, e.g., MyInitializeFunc<dim> must be specified as follows: "
            "this->attach_initial_condition(std::make_shared<MyInitializeFunc<dim>>(), "
            "'level_set') "));
        VectorType solution_level_set;
        scratch_data->initialize_dof_vector(solution_level_set);

        auto ic = base_in->get_initial_condition("level_set");

        dealii::VectorTools::project(scratch_data->get_mapping(),
                                     dof_handler,
                                     constraints,
                                     scratch_data->get_quadrature(),
                                     *ic,
                                     solution_level_set);

        solution_level_set.update_ghost_values();

        /*
         *    initialize the reinitialization operation class
         */

        if (base_in->parameters.reinit.implementation == "meltpooldg")
          {
            reinit_operation = std::make_shared<ReinitializationOperation<dim>>();

            reinit_operation->initialize(scratch_data,
                                         solution_level_set,
                                         base_in->parameters,
                                         reinit_dof_idx,
                                         reinit_quad_idx,
                                         normal_dof_idx);
          }
#ifdef MELT_POOL_DG_WITH_ADAFLO
        else if (base_in->parameters.reinit.implementation == "adaflo")
          {
            AssertThrow(base_in->parameters.reinit.solver.do_matrix_free, ExcNotImplemented());

            reinit_operation = std::make_shared<ReinitializationOperationAdaflo<dim>>(
              *scratch_data,
              reinit_dof_idx,
              reinit_quad_idx,
              normal_dof_idx, // normal vec @todo
              solution_level_set,
              base_in->parameters);
          }
#endif
        else
          AssertThrow(false, ExcNotImplemented());
      }

      void
      setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in)
      {
        /*
         *  setup DoFHandler
         */
#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
        if (base_in->parameters.base.do_simplex)
          dof_handler.distribute_dofs(Simplex::FE_P<dim>(base_in->parameters.base.degree));
        else
#endif
          dof_handler.distribute_dofs(FE_Q<dim>(base_in->parameters.base.degree));

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

        /*
         *  create the matrix-free object
         */
        scratch_data->build();

        if (reinit_operation)
          reinit_operation->reinit();

        /*
         *  initialize postprocessor
         */
        post_processor =
          std::make_shared<Postprocessor<dim>>(scratch_data->get_mpi_comm(reinit_dof_idx),
                                               base_in->parameters.paraview,
                                               scratch_data->get_mapping(),
                                               scratch_data->get_triangulation(reinit_dof_idx));
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
          locally_relevant_solution.reinit(scratch_data->get_partitioner());
          locally_relevant_solution.copy_locally_owned_data_from(reinit_operation->get_level_set());
          constraints.distribute(locally_relevant_solution);
          locally_relevant_solution.update_ghost_values();

          KellyErrorEstimator<dim>::estimate(scratch_data->get_dof_handler(),
                                             scratch_data->get_face_quadrature(),
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
          reinit_operation->attach_vectors(vectors);
        };

        const auto post = [&]() {
          constraints.distribute(reinit_operation->get_level_set());

          VectorType temp(reinit_operation->get_level_set());
          temp.copy_locally_owned_data_from(reinit_operation->get_level_set());
          reinit_operation->update_initial_solution(temp);
        };

        const auto setup_dof_system = [&]() { this->setup_dof_system(base_in); };

        refine_grid<dim, VectorType>(mark_cells_for_refinement,
                                     attach_vectors,
                                     post,
                                     setup_dof_system,
                                     base_in->parameters.amr,
                                     dof_handler);
      }

      /*
       *  Creating paraview output
       */
      void
      output_results(const unsigned int time_step) const
      {
        post_processor->process(time_step, [&](DataOut<dim> &data_out) {
          reinit_operation->attach_output_vectors(data_out);
        });
      }

    private:
      DoFHandler<dim>           dof_handler;
      AffineConstraints<double> constraints;

      std::shared_ptr<ScratchData<dim>>                   scratch_data;
      TimeIterator<double>                                time_iterator;
      std::shared_ptr<ReinitializationOperationBase<dim>> reinit_operation;
      unsigned int                                        reinit_dof_idx;
      unsigned int                                        normal_dof_idx;
      unsigned int                                        reinit_quad_idx;

      std::shared_ptr<Postprocessor<dim>> post_processor;
    };
  } // namespace Reinitialization
} // namespace MeltPoolDG
