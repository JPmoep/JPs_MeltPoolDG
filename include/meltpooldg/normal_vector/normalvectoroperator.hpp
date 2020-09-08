#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
// interface class
//#include "matrixfreeoperator.hpp"

using namespace dealii;

namespace MeltPoolDG
{
  namespace LevelSetMatrixFree
  {

    template<int dim, int degree, typename number = double>
    class NormalVectorOperator // @ interface to be added : public MatrixFreeOperator<number, BlockVectorType, BlockVectorType>
    {
      public:
        using VectorType = LinearAlgebra::distributed::Vector<number>;
        using BlockVectorType = LinearAlgebra::distributed::BlockVector<number>;
        using VectorizedArrayType = VectorizedArray<number>;
        using vector = Tensor<1, dim, VectorizedArray<number>>;
        using scalar = VectorizedArray<number>;
        
        NormalVectorOperator
          ( const MatrixFree<dim, number, VectorizedArrayType> &matrix_free,
            const double damping_in )
          : matrix_free( matrix_free )
          , damping(      damping_in )
          {}

        void
        vmult(BlockVectorType & dst,
              const BlockVectorType & src) const
        {
          const int n_q_points_1d = degree+1; // @ todo: not hard code
          
          FEEvaluation<dim, degree, n_q_points_1d, dim, number>   normal(      matrix_free );

          matrix_free.template cell_loop<BlockVectorType, BlockVectorType>( [&] 
            (const auto&, auto& dst, const auto& src, auto cell_range) {
              for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
              {
                normal.reinit(cell);
              /*
               * @ bug? --> the following call yield a compilation error
               */
                //normal.gather_evaluate(src, true, true);
              /* current work around */
                normal.read_dof_values(src);
                normal.evaluate(true,true,false);

                for (unsigned int q_index=0; q_index<normal.n_q_points; ++q_index)
                {
                    normal.submit_value(              normal.get_value(    q_index ), q_index);
                    normal.submit_gradient( damping * normal.get_gradient( q_index ), q_index );
                  }
                  /*
                  * @ bug? --> the following call yield a compilation error
                  */
                  //normal_comp.integrate_scatter(true, true, dst);
              /* current work around */
                  normal.integrate(true, true);
                  normal.distribute_local_to_global(dst);
                }
              },
              dst, 
              src, 
              true );
        }

        void
        create_rhs(BlockVectorType & dst,
                   const VectorType & src) const
        {
          const int n_q_points_1d = degree+1;

          FEEvaluation<dim, degree, n_q_points_1d, dim, number>   normal_vector( matrix_free );
          FEEvaluation<dim, degree, n_q_points_1d, 1, number>     levelset( matrix_free );

          matrix_free.template cell_loop<BlockVectorType, VectorType>(
            [&](const auto &, auto &dst, const auto &src, auto macro_cells) {
              for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
              {
                normal_vector.reinit(cell);
                levelset.reinit(cell);

                levelset.gather_evaluate(src, false, true);
                for (unsigned int q_index = 0; q_index < normal_vector.n_q_points; ++q_index)
                  normal_vector.submit_value( levelset.get_gradient(q_index), q_index );

                normal_vector.integrate(true, false);
                normal_vector.distribute_local_to_global(dst);
                /*
                 * @ bug? --> the following call yield a compilation error
                 */
                //normal_vector.integrate_scatter(true, false, dst);
              }
            },
            dst,
            src,
            true);
        }

        void
        initialize_dof_vector(VectorType &dst) const
        {
          matrix_free.initialize_dof_vector(dst);
        }
        
        void
        initialize_dof_vector(BlockVectorType &dst) const
        {
          dst.reinit(dim);
          for (unsigned int d=0; d<dim; ++d)
            matrix_free.initialize_dof_vector(dst.block(d));
        }


        private:
          const MatrixFree<dim, number, VectorizedArrayType> &matrix_free;
          double damping; 

      };
    }   // LevelSetMatrixFree
} // MeltPoolDG
