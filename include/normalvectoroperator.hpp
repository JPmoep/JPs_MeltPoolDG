// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

using namespace dealii;

namespace LevelSetParallel
{
  namespace LevelSetMatrixFree
  {

    template<int dim, int degree, typename number = double>
    class NormalVectorOperator
    {
        public:
          typedef LinearAlgebra::distributed::Vector<number>       VectorType;
          typedef LinearAlgebra::distributed::BlockVector<number>  BlockVectorType;
          typedef VectorizedArray<number>                          VectorizedArrayType;
          typedef Tensor<1, dim, VectorizedArray<number>>          vector;
          typedef VectorizedArray<number>                          scalar;
          
          NormalVectorOperator
            (const MatrixFree<dim, number, VectorizedArrayType> &matrix_free,
             const double damping_in)
            : matrix_free( matrix_free )
            , damping(         damping_in )
            {}

          void
          vmult(BlockVectorType & dst,
                const BlockVectorType & src) const
          {
            const int n_q_points_1d = degree+1;
            
            FEEvaluation<dim, degree, n_q_points_1d, dim, number>   normal_comp(      matrix_free );

            matrix_free.template cell_loop<BlockVectorType, BlockVectorType>( [&] 
              (const auto&, auto& dst, const auto& src, auto cell_range) {
                for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
                {
                  normal_comp.reinit(cell);
                /*
                 * @ bug? --> the following call yield a compilation error
                 */
                  //normal_comp.gather_evaluate(src, true, true);
                  normal_comp.read_dof_values(src);
                  normal_comp.evaluate(true,true);
                  for (unsigned int q_index=0; q_index<normal_comp.n_q_points; q_index++)
                  {
                      normal_comp.submit_value(              normal_comp.get_value(    q_index ), q_index);
                      normal_comp.submit_gradient( damping * normal_comp.get_gradient( q_index ), q_index);
                  }
                  /*
                  * @ bug? --> the following call yield a compilation error
                  */
                  //normal_comp.integrate_scatter(true, true, dst);
                  normal_comp.integrate(true, true);
                  normal_comp.distribute_local_to_global(dst);
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

        private:
          const MatrixFree<dim, number, VectorizedArrayType> &matrix_free;
          double damping; 

      };

    }   // LevelSetMatrixFree
} // LevelSetParallel
