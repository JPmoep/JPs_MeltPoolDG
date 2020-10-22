/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
// DoFTools
#include <deal.II/dofs/dof_tools.h>
// MeltPoolDG
#include <meltpooldg/advection_diffusion/advection_diffusion_operator.hpp>
#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/utilities/linearsolve.hpp>
#include <meltpooldg/utilities/utilityfunctions.hpp>

namespace MeltPoolDG
{
  namespace AdvectionDiffusion
  {
    using namespace dealii;

    template <int dim, int comp = 0, int comp_hanging_node_constraints = comp + 1>
    class AdvectionDiffusionOperation
    {
    private:
      using VectorType       = LinearAlgebra::distributed::Vector<double>;
      using BlockVectorType  = LinearAlgebra::distributed::BlockVector<double>;
      using SparseMatrixType = TrilinosWrappers::SparseMatrix;

    public:
      /*
       *    This is the primary solution variable of this module, which will be also publically
       *    accessible for output_results.
       */
      VectorType solution_advected_field;
      /*
       *  All the necessary parameters are stored in this struct.
       */
      AdvectionDiffusionData<double> advec_diff_data;

      AdvectionDiffusionOperation() = default;

      void
      initialize(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
                 const VectorType &                             solution_advected_field_in,
                 const Parameters<double> &                     data_in)
      {
        scratch_data = scratch_data_in;
        /*
         *  set the advection diffusion data
         */
        advec_diff_data = data_in.advec_diff;
        /*
         *  set the initial solution of the advected field
         */
        scratch_data->initialize_dof_vector(solution_advected_field, comp);
        solution_advected_field.copy_locally_owned_data_from(solution_advected_field_in);
        solution_advected_field.update_ghost_values();
        /*
         *  set the parameters for the advection_diffusion problem
         */
        set_advection_diffusion_parameters(data_in);
      }


      void
      solve(const double dt, const BlockVectorType &advection_velocity)
      {
        create_operator(advection_velocity);

        VectorType src, rhs;

        scratch_data->initialize_dof_vector(src);
        scratch_data->initialize_dof_vector(rhs);

        advec_diff_operator->set_time_increment(dt);

        int iter = 0;

        if (advec_diff_data.do_matrix_free)
          {
            /*
             * apply dirichlet boundary values
             */
            auto advec_diff_operator_no_bc = std::make_unique<
              AdvectionDiffusionOperator<dim, comp_hanging_node_constraints, double>>(
              *scratch_data, advection_velocity, advec_diff_data);
            advec_diff_operator_no_bc->set_time_increment(dt);
            advec_diff_operator_no_bc->create_rhs_and_apply_dirichlet_mf(rhs,
                                                                         solution_advected_field,
                                                                         *scratch_data,
                                                                         comp /*dof_idx*/);
            /*
             * solve linear system A*u_0 = f-A*u_D
             */
            iter = LinearSolve<VectorType, SolverGMRES<VectorType>, OperatorBase<double>>::solve(
              *advec_diff_operator, src, rhs);
          }
        else
          {
            //@todo: which preconditioner?
            // TrilinosWrappers::PreconditionAMG preconditioner;
            // TrilinosWrappers::PreconditionAMG::AdditionalData data;

            // preconditioner.initialize(system_matrix, data);
            advec_diff_operator->assemble_matrixbased(solution_advected_field,
                                                      advec_diff_operator->system_matrix,
                                                      rhs);
            iter = LinearSolve<VectorType, SolverGMRES<VectorType>, SparseMatrixType>::solve(
              advec_diff_operator->system_matrix, src, rhs);
          }

        scratch_data->get_constraint(comp).distribute(src);

        solution_advected_field = src;
        solution_advected_field.update_ghost_values();

        if (advec_diff_data.do_print_l2norm)
          {
            const ConditionalOStream &pcout = scratch_data->get_pcout();
            pcout << "| GMRES: i=" << std::setw(5) << std::left << iter;
            pcout << "\t |ϕ|2 = " << std::setw(15) << std::left << std::setprecision(10)
                  << solution_advected_field.l2_norm() << std::endl;
          }
      }

    private:
      void
      set_advection_diffusion_parameters(const Parameters<double> &data_in)
      {
        advec_diff_data = data_in.advec_diff;
      }

      void
      create_operator(const BlockVectorType &advection_velocity)
      {
        advec_diff_operator =
          std::make_unique<AdvectionDiffusionOperator<dim, comp, double>>(*scratch_data,
                                                                          advection_velocity,
                                                                          advec_diff_data);

        /*
         *  In case of a matrix-based simulation, setup the distributed sparsity pattern and
         *  apply it to the system matrix. This functionality is part of the OperatorBase class.
         */
        if (!advec_diff_data.do_matrix_free)
          advec_diff_operator->initialize_matrix_based<dim, comp>(*scratch_data);
      }

    private:
      std::shared_ptr<const ScratchData<dim>> scratch_data;
      /*
       *  This pointer will point to your user-defined advection_diffusion operator.
       */
      std::unique_ptr<OperatorBase<double>> advec_diff_operator;
    };
  } // namespace AdvectionDiffusion
} // namespace MeltPoolDG
