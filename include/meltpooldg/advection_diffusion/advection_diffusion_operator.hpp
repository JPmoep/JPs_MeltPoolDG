/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
// MeltPoolDG
#include <meltpooldg/advection_diffusion/advection_diffusion_operation.hpp>
#include <meltpooldg/interface/operator_base.hpp>

namespace MeltPoolDG
{
  namespace AdvectionDiffusion
  {
    using namespace dealii;

    template <int dim, int comp = 0, typename number = double>
    class AdvectionDiffusionOperator
      : public OperatorBase<number,
                            LinearAlgebra::distributed::Vector<number>,
                            LinearAlgebra::distributed::Vector<number>>
    {
    private:
      using VectorType          = LinearAlgebra::distributed::Vector<number>;
      using BlockVectorType     = LinearAlgebra::distributed::BlockVector<number>;
      using SparseMatrixType    = TrilinosWrappers::SparseMatrix;
      using VectorizedArrayType = VectorizedArray<number>;
      using vector              = Tensor<1, dim, VectorizedArray<number>>;
      using scalar              = VectorizedArray<number>;

    public:
      // clang-format off
    AdvectionDiffusionOperator( const ScratchData<dim>&       scratch_data_in, 
                                const TensorFunction<1,dim>&  advection_velocity_in,
                                const AdvectionDiffusionData<number>& data_in )
    : scratch_data        ( scratch_data_in )
    , advection_velocity  ( advection_velocity_in )
    , data                ( data_in               )
    {
    }
      // clang-format on

      /*
       *    this is the matrix-based implementation of the rhs and the matrix
       *    @todo: this could be improved by using the WorkStream functionality of dealii
       */

      void
      assemble_matrixbased(const VectorType &advected_field_old,
                           SparseMatrixType &matrix,
                           VectorType &      rhs) const override
      {
        AssertThrow(data.diffusivity >= 0.0,
                    ExcMessage("Advection diffusion operator: diffusivity is smaller than zero!"));
        advected_field_old.update_ghost_values();

        const auto &       mapping = scratch_data.get_mapping();
        FEValues<dim>      fe_values(mapping,
                                scratch_data.get_matrix_free().get_dof_handler(comp).get_fe(),
                                scratch_data.get_matrix_free().get_quadrature(comp),
                                update_values | update_gradients | update_quadrature_points |
                                  update_JxW_values);
        const unsigned int dofs_per_cell = scratch_data.get_n_dofs_per_cell();

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double>     cell_rhs(dofs_per_cell);

        const unsigned int n_q_points = fe_values.get_quadrature().size();

        std::vector<double>                  phi_at_q(n_q_points);
        std::vector<Tensor<1, dim>>          grad_phi_at_q(n_q_points, Tensor<1, dim>());
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        rhs    = 0.0;
        matrix = 0.0;

        Tensor<1, dim> a;

        for (const auto &cell :
             scratch_data.get_matrix_free().get_dof_handler(comp).active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell_matrix = 0;
              cell_rhs    = 0;

              fe_values.reinit(cell);
              fe_values.get_function_values(advected_field_old, phi_at_q);
              fe_values.get_function_gradients(advected_field_old, grad_phi_at_q);

              for (const unsigned int q_index : fe_values.quadrature_point_indices())
                {
                  auto qCoord = fe_values.get_quadrature_points()[q_index];
                  a           = advection_velocity.value(qCoord);

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          auto velocity_grad_phi_j = a * fe_values.shape_grad(j, q_index);
                          // clang-format off
                          cell_matrix( i, j ) += 
                                      ( fe_values.shape_value( i, q_index) 
                                         * 
                                         fe_values.shape_value( j, q_index) 
                                         +
                                         data.theta * this->d_tau * ( data.diffusivity * 
                                                               fe_values.shape_grad( i, q_index) * 
                                                               fe_values.shape_grad( j, q_index) +
                                                               fe_values.shape_value( i, q_index) ) *
                                                               velocity_grad_phi_j 
                                      ) * fe_values.JxW(q_index);
                          // clang-format on
                        }

                      // clang-format off
                      cell_rhs( i ) +=
                        (  fe_values.shape_value( i, q_index) * phi_at_q[q_index]
                            - 
                           ( 1. - data.theta ) * this->d_tau * 
                             (
                               data.diffusivity * fe_values.shape_grad( i, q_index) * grad_phi_at_q[q_index]
                               +
                               a * grad_phi_at_q[q_index] * fe_values.shape_value(  i, q_index)
                             )
                          ) * fe_values.JxW(q_index) ;
                      // clang-format on
                    }
                } // end gauss

              // assembly
              cell->get_dof_indices(local_dof_indices);
              scratch_data.get_constraint(comp).distribute_local_to_global(
                cell_matrix, cell_rhs, local_dof_indices, matrix, rhs);
            }

        matrix.compress(VectorOperation::add);
        rhs.compress(VectorOperation::add);
      }

      /*
       *    matrix-free implementation  --> @todo -- wip!!
       *
       */

      // void
      // vmult(VectorType & dst,
      // const VectorType & src) const override
      //{
      //}


      // void
      // create_rhs(VectorType & dst,
      // const VectorType & src) const override
      //{
      //}

    private:
      const ScratchData<dim> &              scratch_data;
      const TensorFunction<1, dim> &        advection_velocity;
      const AdvectionDiffusionData<number> &data;
    };
  } // namespace AdvectionDiffusion
} // namespace MeltPoolDG
