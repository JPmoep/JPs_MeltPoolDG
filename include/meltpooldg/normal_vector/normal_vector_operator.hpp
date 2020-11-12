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
#include <meltpooldg/utilities/fe_integrator.hpp>

using namespace dealii;

namespace MeltPoolDG
{
  namespace NormalVector
  {
    template <int dim, typename number = double>
    class NormalVectorOperator
      : public OperatorBase<number,
                            LinearAlgebra::distributed::BlockVector<number>,
                            LinearAlgebra::distributed::Vector<number>>
    {
    public:
      using VectorType          = LinearAlgebra::distributed::Vector<number>;
      using BlockVectorType     = LinearAlgebra::distributed::BlockVector<number>;
      using VectorizedArrayType = VectorizedArray<number>;
      using SparseMatrixType    = TrilinosWrappers::SparseMatrix;

      NormalVectorOperator(const ScratchData<dim> &scratch_data_in,
                           const double            damping_in,
                           const unsigned int      dof_idx_in,
                           const unsigned int      quad_idx_in)
        : scratch_data(scratch_data_in)
        , damping(damping_in)
      {
        this->reset_indices(dof_idx_in, quad_idx_in);
      }

      void
      assemble_matrixbased(const VectorType &level_set_in,
                           SparseMatrixType &matrix,
                           BlockVectorType & rhs) const override
      {
        level_set_in.update_ghost_values();

        FEValues<dim> fe_values(scratch_data.get_mapping(),
                                scratch_data.get_dof_handler(this->dof_idx).get_fe(),
                                scratch_data.get_quadrature(this->quad_idx),
                                update_values | update_gradients | update_quadrature_points |
                                  update_JxW_values);

        const unsigned int dofs_per_cell = scratch_data.get_n_dofs_per_cell(this->dof_idx);

        FullMatrix<double>                   normal_cell_matrix(dofs_per_cell, dofs_per_cell);
        std::vector<Vector<double>>          normal_cell_rhs(dim, Vector<double>(dofs_per_cell));
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        const unsigned int n_q_points = fe_values.get_quadrature().size();

        std::vector<Tensor<1, dim>> normal_at_q(n_q_points, Tensor<1, dim>());

        matrix = 0.0;
        rhs    = 0.0;

        for (const auto &cell : scratch_data.get_dof_handler(this->dof_idx).active_cell_iterators())
          if (cell->is_locally_owned())
            {
              fe_values.reinit(cell);
              cell->get_dof_indices(local_dof_indices);

              normal_cell_matrix = 0.0;
              for (auto &normal_cell : normal_cell_rhs)
                normal_cell = 0.0;

              fe_values.get_function_gradients(
                level_set_in, normal_at_q); // compute normals from level set solution at tau=0
              for (const unsigned int q_index : fe_values.quadrature_point_indices())
                {
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      const double         phi_i      = fe_values.shape_value(i, q_index);
                      const Tensor<1, dim> grad_phi_i = fe_values.shape_grad(i, q_index);

                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          const double         phi_j      = fe_values.shape_value(j, q_index);
                          const Tensor<1, dim> grad_phi_j = fe_values.shape_grad(j, q_index);

                          //clang-format off
                          normal_cell_matrix(i, j) +=
                            (phi_i * phi_j + damping * grad_phi_i * grad_phi_j) *
                            fe_values.JxW(q_index);
                          //clang-format on
                        }

                      for (unsigned int d = 0; d < dim; ++d)
                        {
                          //clang-format off
                          normal_cell_rhs[d](i) +=
                            phi_i * normal_at_q[q_index][d] * fe_values.JxW(q_index);
                          //clang-format on
                        }
                    }
                }

              // assembly
              cell->get_dof_indices(local_dof_indices);

              scratch_data.get_constraint(this->dof_idx)
                .distribute_local_to_global(normal_cell_matrix, local_dof_indices, matrix);
              for (unsigned int d = 0; d < dim; ++d)
                scratch_data.get_constraint(this->dof_idx)
                  .distribute_local_to_global(normal_cell_rhs[d], local_dof_indices, rhs.block(d));

            } // end of cell loop
        matrix.compress(VectorOperation::add);
        rhs.compress(VectorOperation::add);
      }

      /*
       *  matrix-free utility
       */

      void
      vmult(BlockVectorType &dst, const BlockVectorType &src) const override
      {
        scratch_data.get_matrix_free().template cell_loop<BlockVectorType, BlockVectorType>(
          [&](const auto &, auto &dst, const auto &src, auto cell_range) {
            FECellIntegrator<dim, dim, number> normal(scratch_data.get_matrix_free(),
                                                      this->dof_idx,
                                                      this->quad_idx);
            for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
              {
                normal.reinit(cell);
                normal.gather_evaluate(src, true, true);

                for (unsigned int q_index = 0; q_index < normal.n_q_points; ++q_index)
                  {
                    normal.submit_value(normal.get_value(q_index), q_index);
                    normal.submit_gradient(damping * normal.get_gradient(q_index), q_index);
                  }

                normal.integrate_scatter(true, true, dst);
              }
          },
          dst,
          src,
          true);
      }

      void
      create_rhs(BlockVectorType &dst, const VectorType &src) const override
      {
        FECellIntegrator<dim, dim, number> normal_vector(scratch_data.get_matrix_free(),
                                                         this->dof_idx,
                                                         this->quad_idx);
        FECellIntegrator<dim, 1, number>   level_set(scratch_data.get_matrix_free(),
                                                   this->dof_idx,
                                                   this->quad_idx);

        scratch_data.get_matrix_free().template cell_loop<BlockVectorType, VectorType>(
          [&](const auto &, auto &dst, const auto &src, auto macro_cells) {
            for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
              {
                normal_vector.reinit(cell);

                level_set.reinit(cell);
                level_set.read_dof_values_plain(src);
                level_set.evaluate(false, true);

                for (unsigned int q_index = 0; q_index < normal_vector.n_q_points; ++q_index)
                  normal_vector.submit_value(level_set.get_gradient(q_index), q_index);

                normal_vector.integrate_scatter(true, false, dst);
              }
          },
          dst,
          src,
          true);
      }

      static void
      get_unit_normals_at_quadrature(const FEValues<dim> &        fe_values,
                                     const BlockVectorType &      normal_vector_field_in,
                                     std::vector<Tensor<1, dim>> &unit_normal_at_quadrature)
      {
        for (unsigned int d = 0; d < dim; ++d)
          {
            std::vector<double> temp(unit_normal_at_quadrature.size());
            fe_values.get_function_values(normal_vector_field_in.block(d),
                                          temp); // compute normals from level set solution at tau=0
            for (const unsigned int q_index : fe_values.quadrature_point_indices())
              unit_normal_at_quadrature[q_index][d] = temp[q_index];
          }
        for (auto &n : unit_normal_at_quadrature)
          n /= n.norm(); //@todo: add exception if norm is zero
      }

    private:
      const ScratchData<dim> &scratch_data;

      double damping;
    };
  } // namespace NormalVector
} // namespace MeltPoolDG
