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
class ReinitializationOperator
{
    public:
    typedef LinearAlgebra::distributed::Vector<number>  VectorType;
    typedef VectorizedArray<number>     VectorizedArrayType ;


    ReinitializationOperator
      ()
      : matrix_free( MatrixFree<dim,number>())
      {}

    void
    vmult(VectorType & destination,
          const VectorType & source) const
    {
      const int n_q_points = degree+1;
      
      FEEvaluation<dim, degree, n_q_points, 1, number> fe_eval(matrix_free);

      matrix_free.template cell_loop<VectorType, VectorType>( [&] 
        (const auto&, auto& dest, const auto& src, auto &cell_range) {
          for (auto cell = cell_range.first; cell<cell_range.second; ++cell)
          {
            fe_eval.reinit(cell);

            // #1
            fe_eval.gather_evaluate(src, true, false);
            for (unsigned int q_index=0; q_index<fe_eval.n_q_points; q_index++)
            {
                // #2
                const double uq = fe_eval.get_value(q_index);
                fe_eval.submit_value(uq, q_index);
            }
            // #3
            fe_eval.integrate_scatter(true,false,dest);
          }
        },
        destination, 
        source, 
        true );

    }

    void
    initialize_dof_vector(VectorType &dst) const
    {
      matrix_free.initialize_dof_vector(dst);
    }

    private:
      const MatrixFree<dim, number, VectorizedArrayType> &matrix_free;

};
}   // LevelSetMatrixFree
} // LevelSetParallel
