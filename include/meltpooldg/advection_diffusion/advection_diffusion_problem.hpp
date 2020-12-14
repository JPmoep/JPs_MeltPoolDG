/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/index_set.h>

#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/generic_linear_algebra.h>

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
            output_results(time_iterator.get_current_time_step_number(), base_in->parameters);
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
      initialize(std::shared_ptr<SimulationBase<dim>> base_in)
      {
        /*
         *  setup scratch data
         */
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
        unsigned int quad_idx = 0;
#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
        if (base_in->parameters.base.do_simplex)
          quad_idx = scratch_data->attach_quadrature(
            Simplex::QGauss<dim>(base_in->parameters.base.n_q_points_1d));
        else
#endif
          quad_idx =
            scratch_data->attach_quadrature(QGauss<1>(base_in->parameters.base.n_q_points_1d));
        /*
         *  setup DoFHandler
         */
        dof_handler.reinit(*base_in->triangulation);
        dof_handler_velocity.reinit(*base_in->triangulation);
#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
        if (base_in->parameters.base.do_simplex)
          {
            dof_handler.distribute_dofs(Simplex::FE_P<dim>(base_in->parameters.base.degree));
            dof_handler_velocity.distribute_dofs(
              FESystem(Simplex::FE_P<dim>(base_in->parameters.base.degree), dim));
          }
        else
#endif
          {
            dof_handler.distribute_dofs(FE_Q<dim>(base_in->parameters.base.degree));
            dof_handler_velocity.distribute_dofs(
              FESystem<dim,dim>(FE_Q<dim>(base_in->parameters.base.degree), dim));
          }

        dof_idx         = scratch_data->attach_dof_handler(dof_handler);
        dof_no_bc_idx   = scratch_data->attach_dof_handler(dof_handler);
        dof_zero_bc_idx = scratch_data->attach_dof_handler(dof_handler);

        const int dof_idx_velocity = scratch_data->attach_dof_handler(dof_handler_velocity);

        /*
         *  create the partititioning
         */
        scratch_data->create_partitioning();
        /*
         *  make hanging nodes and dirichlet constraints (Note: at the moment no time-dependent
         *  dirichlet constraints are supported)
         */
        hanging_node_constraints.clear();
        hanging_node_constraints.reinit(scratch_data->get_locally_relevant_dofs(dof_idx));
        DoFTools::make_hanging_node_constraints(dof_handler, hanging_node_constraints);
        hanging_node_constraints.close();

        hanging_node_constraints_with_zero_dirichlet.clear();
        hanging_node_constraints_with_zero_dirichlet.reinit(
          scratch_data->get_locally_relevant_dofs(dof_idx));
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
          scratch_data->get_locally_relevant_dofs(dof_idx_velocity));
        DoFTools::make_hanging_node_constraints(dof_handler_velocity,
                                                hanging_node_constraints_velocity);
        hanging_node_constraints_velocity.close();

        constraints.clear();
        constraints.reinit(scratch_data->get_locally_relevant_dofs());
        constraints.merge(hanging_node_constraints,
                          AffineConstraints<double>::MergeConflictBehavior::left_object_wins);
        for (const auto &bc : base_in->get_dirichlet_bc(
               "advection_diffusion")) // @todo: add name of bc at a more central place
          {
            dealii::VectorTools::interpolate_boundary_values(
              scratch_data->get_mapping(), dof_handler, bc.first, *bc.second, constraints);
          }
        constraints.close();

        scratch_data->attach_constraint_matrix(constraints);
        scratch_data->attach_constraint_matrix(hanging_node_constraints);
        scratch_data->attach_constraint_matrix(hanging_node_constraints_with_zero_dirichlet);
        scratch_data->attach_constraint_matrix(hanging_node_constraints_velocity);
        /*
         *  create the matrix-free object
         */
        scratch_data->build();
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
        AssertThrow(base_in->get_advection_field("advection_diffusion"),
                    ExcMessage(
                      " It seems that your SimulationBase object does not contain "
                      "a valid advection velocity. A shared_ptr to your advection velocity "
                      "function, e.g., AdvectionFunc<dim> must be specified as follows: "
                      "this->attach_advection_field(std::make_shared<AdvecFunc<dim>>(), "
                      "'advection_diffusion') "));
        compute_advection_velocity(*base_in->get_advection_field("advection_diffusion"));
        if (base_in->parameters.advec_diff.implementation == "meltpooldg")
          {
            advec_diff_operation = std::make_shared<AdvectionDiffusionOperation<dim>>();

            advec_diff_operation->initialize(scratch_data,
                                             initial_solution,
                                             base_in->parameters,
                                             dof_idx,
                                             dof_no_bc_idx,
                                             quad_idx,
                                             dof_no_bc_idx);
          }
#ifdef MELT_POOL_DG_WITH_ADAFLO
        else if (base_in->parameters.advec_diff.implementation == "adaflo")
          {
            AssertThrow(base_in->parameters.advec_diff.do_matrix_free, ExcNotImplemented());
            advec_diff_operation =
              std::make_shared<MeltPoolDG::AdvectionDiffusionAdaflo::AdafloWrapper<dim>>(
                *scratch_data,
                dof_zero_bc_idx,
                quad_idx,
                dof_idx_velocity,
                initial_solution,
                advection_velocity,
                base_in);
          }
#endif
        else
          AssertThrow(false, ExcNotImplemented());
      }

      void
      compute_advection_velocity(TensorFunction<1, dim> &advec_func)
      {
        scratch_data->initialize_dof_vector(advection_velocity);
        /*
         *  set the current time to the advection field function
         */
        advec_func.set_time(time_iterator.get_current_time());
        /*
         *  work around to interpolate a vector-valued quantity on a scalar DoFHandler
         *  @todo: could be shifted to a utility function
         */
        for (auto d = 0; d < dim; ++d)
          {
            dealii::VectorTools::interpolate(scratch_data->get_mapping(),
                                             scratch_data->get_dof_handler(),
                                             ScalarFunctionFromFunctionObject<dim>(
                                               [&](const Point<dim> &p) {
                                                 return advec_func.value(p)[d];
                                               }),
                                             advection_velocity.block(d));
          }
        advection_velocity.update_ghost_values();
      }

      void
      output_results(const unsigned int time_step, const Parameters<double> &parameters)
      {
        if (parameters.paraview.do_output)
          {
            const MPI_Comm mpi_communicator = scratch_data->get_mpi_comm();

            // advec_diff_operation.solution_advected_field.update_ghost_values();
            advection_velocity.update_ghost_values();
            /*
             *  output advected field
             */
            DataOut<dim> data_out;
            data_out.attach_dof_handler(dof_handler);
            // data_out.add_data_vector(advec_diff_operation.solution_advected_field,
            //"advected_field");

            /*
             *  output advection velocity
             */
            if (parameters.paraview.print_advection)
              {
                for (auto d = 0; d < dim; ++d)
                  data_out.add_data_vector(dof_handler,
                                           advection_velocity.block(d),
                                           "advection_velocity_" + std::to_string(d));
              }
            /*
             * write data to vtu file
             */
            data_out.build_patches(scratch_data->get_mapping());
            data_out.write_vtu_with_pvtu_record("./",
                                                parameters.paraview.filename,
                                                time_step,
                                                scratch_data->get_mpi_comm(),
                                                parameters.paraview.n_digits_timestep,
                                                parameters.paraview.n_groups);

            // advec_diff_operation.solution_advected_field.zero_out_ghosts();
            advection_velocity.zero_out_ghosts();
            /*
             * write data of boundary -- @todo: move to own utility function
             */
            if (parameters.paraview.print_boundary_id)
              {
                const unsigned int rank    = Utilities::MPI::this_mpi_process(mpi_communicator);
                const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(mpi_communicator);

                const unsigned int n_digits =
                  static_cast<int>(std::ceil(std::log10(std::fabs(n_ranks))));

                std::string filename = "./solution_advection_diffusion_boundary_IDs" +
                                       Utilities::int_to_string(rank, n_digits) + ".vtk";
                std::ofstream output(filename.c_str());

                GridOut           grid_out;
                GridOutFlags::Vtk flags;
                flags.output_cells         = false;
                flags.output_faces         = true;
                flags.output_edges         = false;
                flags.output_only_relevant = false;
                grid_out.set_flags(flags);
                grid_out.write_vtk(scratch_data->get_dof_handler().get_triangulation(), output);
              }
          }
      }

    private:
      DoFHandler<dim>                   dof_handler;
      AffineConstraints<double>         constraints;
      AffineConstraints<double>         hanging_node_constraints;
      AffineConstraints<double>         hanging_node_constraints_with_zero_dirichlet;
      DoFHandler<dim>                   dof_handler_velocity;
      AffineConstraints<double>         hanging_node_constraints_velocity;
      std::shared_ptr<ScratchData<dim>> scratch_data;
      BlockVectorType                   advection_velocity;
      TimeIterator<double>              time_iterator;
      std::shared_ptr<AdvectionDiffusionOperationBase<dim>> advec_diff_operation;

      int dof_idx;
      int dof_no_bc_idx;
      int dof_zero_bc_idx;
    };
  } // namespace AdvectionDiffusion
} // namespace MeltPoolDG
