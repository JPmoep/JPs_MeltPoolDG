/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, August 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
// for using smart pointers
#include <deal.II/base/smartpointer.h>

// MeltPoolDG
#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/normal_vector/normal_vector_operation_base.hpp>
#include <meltpooldg/normal_vector/normal_vector_operator.hpp>
#include <meltpooldg/utilities/linearsolve.hpp>
#include <meltpooldg/utilities/utilityfunctions.hpp>

namespace MeltPoolDG
{
  namespace NormalVector
  {
    using namespace dealii;

    /**
     *  This function calculates the normal vector of the current level set function being
     *  the solution of an intermediate projection step
     *
     *              (w, n_ϕ)  + η_n (∇w, ∇n_ϕ)  = (w,∇ϕ)
     *                      Ω                 Ω            Ω
     *
     *  with test function w, the normal vector n_ϕ, damping parameter η_n and the
     *  level set function ϕ.
     *
     *    !!!!
     *          the normal vector field is not normalized to length one,
     *          it actually represents the gradient of the level set
     *          function
     *    !!!!
     */

    template <int dim>
    class NormalVectorOperation : public NormalVectorOperationBase<dim>
    {
    private:
      using VectorType       = LinearAlgebra::distributed::Vector<double>;
      using BlockVectorType  = LinearAlgebra::distributed::BlockVector<double>;
      using SparseMatrixType = TrilinosWrappers::SparseMatrix;

    public:
      NormalVectorData<double> normal_vector_data;
      /*
       *    This is the primary solution variable of this module, which will be also publically
       *    accessible for output_results.
       */
      BlockVectorType solution_normal_vector;

      NormalVectorOperation() = default;

      void
      initialize(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
                 const Parameters<double> &                     data_in,
                 const unsigned int                             normal_dof_idx_in,
                 const unsigned int                             normal_quad_idx_in,
                 const unsigned int                             ls_dof_idx_in) override
      {
        scratch_data    = scratch_data_in;
        normal_dof_idx  = normal_dof_idx_in;
        normal_quad_idx = normal_quad_idx_in;
        ls_dof_idx      = ls_dof_idx_in;
        /*
         *  initialize normal vector data
         */
        normal_vector_data = data_in.normal_vec;
        /*
         *  initialize normal vector operator
         */
        create_operator();
      }

      void
      reinit() override
      {
        if (!normal_vector_data.do_matrix_free)
          normal_vector_operator->initialize_matrix_based<dim>(*scratch_data);
      }

      void
      solve(const VectorType &solution_levelset_in) override
      {
        BlockVectorType rhs;

        scratch_data->initialize_dof_vector(rhs, normal_dof_idx);
        scratch_data->initialize_dof_vector(solution_normal_vector, normal_dof_idx);

        int iter = 0;

        if (normal_vector_data.do_matrix_free)
          {
            normal_vector_operator->create_rhs(rhs, solution_levelset_in);
            iter = LinearSolve<
              BlockVectorType,
              SolverCG<BlockVectorType>,
              OperatorBase<double, BlockVectorType, VectorType>>::solve(*normal_vector_operator,
                                                                        solution_normal_vector,
                                                                        rhs);
          }
        else
          {
            normal_vector_operator->assemble_matrixbased(solution_levelset_in,
                                                         normal_vector_operator->system_matrix,
                                                         rhs);

            for (unsigned int d = 0; d < dim; ++d)
              iter = LinearSolve<VectorType, SolverCG<VectorType>, SparseMatrixType>::solve(
                normal_vector_operator->system_matrix,
                solution_normal_vector.block(d),
                rhs.block(d));
          }
        for (unsigned int d = 0; d < dim; ++d)
          scratch_data->get_constraint(normal_dof_idx).distribute(solution_normal_vector.block(d));

        if (normal_vector_data.do_print_l2norm)
          {
            const ConditionalOStream &pcout = scratch_data->get_pcout();
            pcout << "| normal vector:         i=" << iter << " \t";
            for (unsigned int d = 0; d < dim; ++d)
              pcout << "|n_" << d << "| = " << std::setprecision(11) << std::setw(15) << std::left
                    << solution_normal_vector.block(d).l2_norm();
            pcout << std::endl;
          }
      }

      const BlockVectorType &
      get_solution_normal_vector() const override
      {
        return solution_normal_vector;
      }

      BlockVectorType &
      get_solution_normal_vector() override
      {
        return solution_normal_vector;
      }

    private:
      /**
       * This function creates the normal vector operator for assembling the system operator (either
       * matrix based or matrix free) and the right handside.
       */
      void
      create_operator()
      {
        // epsilon*h^2
        const double damping_parameter =
          std::pow(scratch_data->get_min_cell_size(normal_dof_idx), 2) *
          normal_vector_data.damping_scale_factor;
        // epsilon*h  
        // const double damping_parameter = scratch_data->get_min_cell_size(normal_dof_idx)*normal_vector_data.damping_scale_factor;
        normal_vector_operator = std::make_unique<NormalVectorOperator<dim>>(
          *scratch_data, damping_parameter, normal_dof_idx, normal_quad_idx, ls_dof_idx);
        /*
         *  In case of a matrix-based simulation, setup the distributed sparsity pattern and
         *  apply it to the system matrix. This functionality is part of the OperatorBase class.
         */
        if (!normal_vector_data.do_matrix_free)
          normal_vector_operator->initialize_matrix_based<dim>(*scratch_data);
      }

    private:
      std::shared_ptr<const ScratchData<dim>> scratch_data;
      /*
       *  This pointer will point to your user-defined normal vector operator.
       */
      std::unique_ptr<OperatorBase<double, BlockVectorType, VectorType>> normal_vector_operator;
      /*
       *  Based on the following indices the correct DoFHandler or quadrature rule from
       *  ScratchData<dim> object is selected. This is important when ScratchData<dim> holds
       *  multiple DoFHandlers, quadrature rules, etc.
       */
      unsigned int normal_dof_idx;
      unsigned int normal_quad_idx;
      unsigned int ls_dof_idx;
    };
  } // namespace NormalVector
} // namespace MeltPoolDG
