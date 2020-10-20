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

#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/numerics/data_out.h>

// MeltPoolDG
#include <meltpooldg/advection_diffusion/advection_diffusion_operation.hpp>
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
            compute_advection_velocity(*base_in->get_advection_field());
            advec_diff_operation.solve(dt, advection_velocity);
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
        scratch_data = std::make_shared<ScratchData<dim>>();
        /*
         *  setup mapping
         */
        const auto mapping = MappingQGeneric<dim>(base_in->parameters.base.degree);
        scratch_data->set_mapping(mapping);
        /*
         *  setup DoFHandler
         */
        FE_Q<dim> fe(base_in->parameters.base.degree);

        dof_handler.initialize(*base_in->triangulation, fe);
        scratch_data->attach_dof_handler(dof_handler);
        scratch_data->attach_dof_handler(dof_handler);
        /*
         *  create the partititioning
         */
        scratch_data->create_partitioning();
        /*
           *  make hanging nodes and dirichlet constraints (Note: at the moment no time-dependent
         *  dirichlet constraints are supported)
         */
        hanging_node_constraints.clear();
        hanging_node_constraints.reinit(scratch_data->get_locally_relevant_dofs());
        DoFTools::make_hanging_node_constraints(dof_handler, hanging_node_constraints);
        hanging_node_constraints.close();
        
        constraints.clear();
        constraints.reinit(scratch_data->get_locally_relevant_dofs());
        constraints.merge(hanging_node_constraints);
        for (const auto &bc : base_in->get_boundary_conditions().dirichlet_bc)
          {
            VectorTools::interpolate_boundary_values(dof_handler,
                                                     bc.first,
                                                     *bc.second,
                                                     constraints);
          }
        constraints.close();

        scratch_data->attach_constraint_matrix(constraints);
        scratch_data->attach_constraint_matrix(hanging_node_constraints);
        /*
         *  create quadrature rule
         */
        QGauss<1> quad_1d_temp(base_in->parameters.base.n_q_points_1d);

        scratch_data->attach_quadrature(quad_1d_temp);
        scratch_data->attach_quadrature(quad_1d_temp);
        /*
         *  create the matrix-free object
         */
        scratch_data->build();
        /*
         *  initialize the time iterator
         */
        time_iterator.initialize(TimeIteratorData<double>{
                    base_in->parameters.advec_diff.start_time,
                    base_in->parameters.advec_diff.end_time,
                    base_in->parameters.advec_diff.time_step_size,
                    10000,
                    false});

        /*
         *  set initial conditions of the levelset function
         */
        VectorType initial_solution;
        scratch_data->initialize_dof_vector(initial_solution);

        VectorTools::project(dof_handler,
                             constraints,
                             scratch_data->get_quadrature(),
                             *base_in->get_field_conditions()->initial_field,
                             initial_solution);

        initial_solution.update_ghost_values();
        /*
         *    initialize the advection-diffusion operation class
         */
        AssertThrow(base_in->get_advection_field(), 
                    ExcMessage(" It seems that your SimulationBase object does not contain "
                               "a valid advection velocity. A shared_ptr to your advection velocity "
                               "function, e.g., AdvectionFunc<dim> must be specified as follows: "
                               "this->field_conditions.advection_field = std::make_shared<AdvectionFunc<dim>>();" 
                              ));
        advec_diff_operation.initialize(scratch_data,
                                        initial_solution,
                                        base_in->parameters);
      }
      
      void
      compute_advection_velocity(TensorFunction<1,dim>& advec_func)
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
          VectorTools::interpolate(scratch_data->get_mapping(),
                                   scratch_data->get_dof_handler(),
                                   ScalarFunctionFromFunctionObject<dim>(
                                     [&](const Point<dim> &p) {
                                       return advec_func.value(p)[d];
                                     }),
                                   advection_velocity.block(d));
          advection_velocity.block(d).update_ghost_values();
        }
      }

      void
      output_results(const unsigned int time_step, const Parameters<double> &parameters)
      {
        if (parameters.paraview.do_output)
          {
            const MPI_Comm mpi_communicator = scratch_data->get_mpi_comm();

            /*
             *  output advected field
             */
            DataOut<dim> data_out;
            data_out.attach_dof_handler(dof_handler);
            data_out.add_data_vector(advec_diff_operation.solution_advected_field,
                                     "advected_field");

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
            data_out.build_patches();
            data_out.write_vtu_with_pvtu_record("./", 
                                                parameters.paraview.filename, 
                                                time_step, 
                                                scratch_data->get_mpi_comm(), 
                                                parameters.paraview.n_digits_timestep, 
                                                parameters.paraview.n_groups);

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
      DoFHandler<dim>                         dof_handler;
      AffineConstraints<double>               constraints;
      AffineConstraints<double>               hanging_node_constraints;
      std::shared_ptr<ScratchData<dim>>       scratch_data;
      BlockVectorType                         advection_velocity;
      TimeIterator<double>                    time_iterator;
      AdvectionDiffusionOperation<dim>        advec_diff_operation;
    };
  } // namespace AdvectionDiffusion
} // namespace MeltPoolDG
