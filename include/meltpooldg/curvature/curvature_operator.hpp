/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
// MeltPoolDG
#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/normal_vector/normal_vector_operator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

using namespace dealii;

namespace MeltPoolDG
{
  namespace Curvature
  {
    /**
     *  This function calculates the curvature of the current level set function being
     *  the solution of an intermediate projection step
     *
     *              (w, κ)   +   η_κ (∇w, ∇κ)  = (w,∇·n_ϕ)
     *                    Ω                  Ω            Ω
     *
     *  with test function w, curvature κ, damping parameter η_κ and the normal to the
     *  level set function n_ϕ.
     *
     */
    template <int dim, typename number = double>
    class CurvatureOperator : public OperatorBase<number,
                                                  LinearAlgebra::distributed::Vector<number>,
                                                  LinearAlgebra::distributed::BlockVector<number>>
    {
    public:
      using VectorType          = LinearAlgebra::distributed::Vector<number>;
      using BlockVectorType     = LinearAlgebra::distributed::BlockVector<number>;
      using VectorizedArrayType = VectorizedArray<number>;
      using SparseMatrixType    = TrilinosWrappers::SparseMatrix;
      using vector              = Tensor<1, dim, VectorizedArray<number>>;
      using scalar              = VectorizedArray<number>;

      // clang-format off
      CurvatureOperator( const ScratchData<dim>& scratch_data_in,
                         const double            damping_in,
                         const unsigned int      curv_dof_idx_in,
                         const unsigned int      curv_quad_idx_in,
                         const unsigned int      normal_dof_idx_in)
      : scratch_data( scratch_data_in )
      , damping     ( damping_in      )
      , curv_dof_idx( curv_dof_idx_in )
      , curv_quad_idx( curv_quad_idx_in )
      , normal_dof_idx( normal_dof_idx_in )
      {
        this->reset_indices(curv_dof_idx_in, curv_quad_idx_in);
      }
      // clang-format on

      void
      assemble_matrixbased(const BlockVectorType &solution_normal_vector_in,
                           SparseMatrixType &     matrix,
                           VectorType &           rhs) const override
      {
        solution_normal_vector_in.update_ghost_values();

        FEValues<dim> curv_values(scratch_data.get_mapping(),
                                  scratch_data.get_dof_handler(curv_dof_idx).get_fe(),
                                  scratch_data.get_quadrature(curv_quad_idx),
                                  update_values | update_gradients | update_quadrature_points |
                                    update_JxW_values);

        FEValues<dim> normal_values(scratch_data.get_mapping(),
                                    scratch_data.get_dof_handler(normal_dof_idx).get_fe(),
                                    scratch_data.get_quadrature(curv_quad_idx),
                                    update_values);

        const unsigned int dofs_per_cell = scratch_data.get_n_dofs_per_cell(curv_dof_idx);

        FullMatrix<double>                   curvature_cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double>                       curvature_cell_rhs(dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        const unsigned int n_q_points = curv_values.get_quadrature().size();

        std::vector<Tensor<1, dim>> normal_at_q(n_q_points, Tensor<1, dim>());

        matrix = 0.0;
        rhs    = 0.0;

        for (const auto &cell : scratch_data.get_dof_handler(curv_dof_idx).active_cell_iterators())
          if (cell->is_locally_owned())
            {
              curv_values.reinit(cell);
              normal_values.reinit(cell);
              cell->get_dof_indices(local_dof_indices);

              curvature_cell_matrix = 0.0;
              curvature_cell_rhs    = 0.0;

              NormalVector::NormalVectorOperator<dim>::get_unit_normals_at_quadrature(
                normal_values, solution_normal_vector_in, normal_at_q);

              for (const unsigned int q_index : curv_values.quadrature_point_indices())
                {
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      const double         phi_i      = curv_values.shape_value(i, q_index);
                      const Tensor<1, dim> grad_phi_i = curv_values.shape_grad(i, q_index);

                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          const double         phi_j      = curv_values.shape_value(j, q_index);
                          const Tensor<1, dim> grad_phi_j = curv_values.shape_grad(j, q_index);

                          curvature_cell_matrix(i, j) +=
                            (phi_i * phi_j + damping * grad_phi_i * grad_phi_j) *
                            curv_values.JxW(q_index);
                        }
                      curvature_cell_rhs(i) +=
                        (grad_phi_i * normal_at_q[q_index] * curv_values.JxW(q_index));
                    }
                }

              // assembly
              cell->get_dof_indices(local_dof_indices);
              scratch_data.get_constraint(curv_dof_idx)
                .distribute_local_to_global(
                  curvature_cell_matrix, curvature_cell_rhs, local_dof_indices, matrix, rhs);

            } // end of cell loop
        matrix.compress(VectorOperation::add);
        rhs.compress(VectorOperation::add);
      }

      /*
       *  matrix-free utility
       */

      void
      vmult(VectorType &dst, const VectorType &src) const override
      {
        scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
          [&](const auto &matrix_free, auto &dst, const auto &src, auto cell_range) {
            FECellIntegrator<dim, 1, number> curvature(matrix_free, curv_dof_idx, curv_quad_idx);
            for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
              {
                curvature.reinit(cell);
                curvature.read_dof_values_plain(src);
                curvature.evaluate(true, true);

                for (unsigned int q_index = 0; q_index < curvature.n_q_points; ++q_index)
                  {
                    curvature.submit_value(curvature.get_value(q_index), q_index);
                    curvature.submit_gradient(damping * curvature.get_gradient(q_index), q_index);
                  }

                curvature.integrate_scatter(true, true, dst);
              }
          },
          dst,
          src,
          true /*zero out dst*/);
      }

      void
      create_rhs(VectorType &dst, const BlockVectorType &src) const override
      {
        scratch_data.get_matrix_free().template cell_loop<VectorType, BlockVectorType>(
          [&](const auto &matrix_free, auto &dst, const auto &src, auto macro_cells) {
            FECellIntegrator<dim, 1, number>   curvature(matrix_free, curv_dof_idx, curv_quad_idx);
            FECellIntegrator<dim, dim, number> normal_vector(matrix_free,
                                                             normal_dof_idx,
                                                             curv_quad_idx);
            for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
              {
                curvature.reinit(cell);

                normal_vector.reinit(cell);
                normal_vector.read_dof_values_plain(src);
                normal_vector.evaluate(true, false);

                for (unsigned int q_index = 0; q_index < curvature.n_q_points; ++q_index)
                  {
                    const auto n_phi =
                      MeltPoolDG::VectorTools::normalize<dim>(normal_vector.get_value(q_index));
                    curvature.submit_gradient(n_phi, q_index);
                  }

                curvature.integrate_scatter(false, true, dst);
              }
          },
          dst,
          src,
          true /*zero out dst*/);
      }

    private:
      const ScratchData<dim> &scratch_data;

      double             damping;
      const unsigned int curv_dof_idx;
      const unsigned int curv_quad_idx;
      const unsigned int normal_dof_idx;
    };
  } // namespace Curvature
} // namespace MeltPoolDG
