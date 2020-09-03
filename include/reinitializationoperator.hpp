// for parallelization
#pragma once
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

using namespace dealii;

namespace MeltPoolDG
{
    namespace LevelSetMatrixFree
    {

template<int dim, int degree, typename number = double>
class ReinitializationOperator
{
    public:
      typedef LinearAlgebra::distributed::Vector<number>       VectorType;
      typedef LinearAlgebra::distributed::BlockVector<number>  BlockVectorType;
      typedef VectorizedArray<number>                          VectorizedArrayType;
      typedef Tensor<1, dim, VectorizedArray<number>>          vector;
      typedef VectorizedArray<number>                          scalar;
      
      ReinitializationOperator
        (const MatrixFree<dim, number, VectorizedArrayType> &matrix_free,
         const double time_increment,
         const double diffusion)
        : matrix_free( matrix_free )
        , d_tau(       time_increment )
        , eps(         diffusion )
        {}

      void
      vmult(VectorType & dst,
            const VectorType & src) const
      {
        AssertThrow(d_tau>0.0, ExcMessage("reinitialization operator: d_tau must be set"));
        AssertThrow(eps>0.0, ExcMessage("reinitialization operator: epsilon must be set"));
        
        const int n_q_points_1d = degree+1;
        
        FEEvaluation<dim, degree, n_q_points_1d, 1, number>   levelset(      matrix_free );
        FEEvaluation<dim, degree, n_q_points_1d, dim, number> normal_vector( matrix_free );

        matrix_free.template cell_loop<VectorType, VectorType>( [&] 
          (const auto&, auto& dst, const auto& src, auto cell_range) {
            for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
            {
              levelset.reinit(cell);
              levelset.gather_evaluate(src, true, true);
              
              normal_vector.reinit(cell);
              normal_vector.read_dof_values(n);
              normal_vector.evaluate(true, false);

              for (unsigned int q_index=0; q_index<levelset.n_q_points; q_index++)
              {
                  const scalar phi =      levelset.get_value(    q_index );
                  
                  const vector grad_phi = levelset.get_gradient( q_index );
                  
                  vector n_phi = normal_vector.get_value( q_index );
                  n_phi /= n_phi.norm();
                  
                  levelset.submit_value(phi, q_index);
                  levelset.submit_gradient(d_tau * eps * scalar_product(grad_phi, n_phi) * n_phi, q_index);
              }

              levelset.integrate_scatter(true, true, dst);
            }
          },
          dst, 
          src, 
          true );
    }


    void
    create_rhs(VectorType & dst,
               const VectorType & src,
               const BlockVectorType& normals) const
    {
      AssertThrow(d_tau>0.0, ExcMessage("reinitialization operator: d_tau must be set"));
      AssertThrow(eps>0.0, ExcMessage("reinitialization operator: epsilon must be set"));
      
      const auto compressive_flux = [&](const auto &phi) 
      {
          return 0.5 * ( make_vectorized_array<number>(1.) - phi * phi );
      };

      const int n_q_points_1d = degree+1;
      
      FEEvaluation<dim, degree, n_q_points_1d, 1, number>   psi(           matrix_free);
      FEEvaluation<dim, degree, n_q_points_1d, dim, number> normal_vector( matrix_free);
  
      matrix_free.template cell_loop<VectorType, VectorType>(
        [&](const auto &, auto &dst, const auto &src, auto macro_cells) {
          for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            psi.reinit(cell);
            psi.gather_evaluate(src, true, true);
            
            normal_vector.reinit(cell);
            normal_vector.read_dof_values(normals);
            normal_vector.evaluate(true, false);

            for (unsigned int q_index = 0; q_index < psi.n_q_points; ++q_index)
            {
              const scalar val = psi.get_value(q_index);
              vector n_phi = normal_vector.get_value(q_index);
                     n_phi /= n_phi.norm();
              psi.submit_gradient( d_tau * compressive_flux(val) * n_phi 
                                   - 
                                   d_tau * eps * scalar_product( psi.get_gradient(q_index), n_phi ) * n_phi, q_index);
            }

            psi.integrate_scatter(false, true, dst);
          }
        },
        dst,
        src,
        true);
    }

    void
    set_normal_vector_field(const BlockVectorType &normal_vector) 
    {
      n.reinit(dim);
      for (unsigned int d=0; d<dim; ++d)
      {
        matrix_free.initialize_dof_vector(n.block(d));
        n.block(d).copy_locally_owned_data_from(normal_vector.block(d));
      }
    }

    void
    initialize_dof_vector(VectorType &dst) const
    {
      matrix_free.initialize_dof_vector(dst);
    }

    void 
    set_time_increment(const double &dt)
    {
      d_tau = dt;
    }

    private:
      const MatrixFree<dim, number, VectorizedArrayType> &matrix_free;
      double d_tau; 
      double eps; 
      BlockVectorType n;

};
}   // namespace LevelSetMatrixFree
} // namespace MeltPoolDG
