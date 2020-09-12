/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
// MeltPoolDG
#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/normal_vector_refactored/normal_vector_operator.hpp>

namespace MeltPoolDG
{
namespace ReinitializationNew
{
using namespace dealii;

template<int dim, int degree, typename number = double, unsigned int comp=0>
class OlssonOperator : public OperatorBase<number, 
                              LinearAlgebra::distributed::Vector<number>, 
                              LinearAlgebra::distributed::Vector<number>>
{
    private:
        using VectorType              = LinearAlgebra::distributed::Vector<number>;          
        using BlockVectorType         = LinearAlgebra::distributed::BlockVector<number>;
        using SparseMatrixType        = TrilinosWrappers::SparseMatrix;
        using VectorizedArrayType     = VectorizedArray<number>;                   
        using vector                  = Tensor<1, dim, VectorizedArray<number>>;                  
        using scalar                  = VectorizedArray<number>;                                  
  public:
      OlssonOperator
        ( const double                                            time_increment,
          const BlockVectorType&                                  n_in,
          const MatrixFree<dim, double, VectorizedArray<double>>& scratch_data_in,
          SmartPointer<const AffineConstraints<number>>           constraints_in,
          const double                                            min_cell_size
        )
        : constraints         ( constraints_in)
        , matrix_free(          scratch_data_in ) 
        , eps                 ( min_cell_size / (std::sqrt(dim) * 2))
        {
          this->set_time_increment(time_increment);
          /*
           *  initialize normal_vector
           */
          set_normal_vector_field(n_in);
        }
      
      /*
       *    this is the matrix-based implementation of the rhs and the system_matrix
       *    @todo: this could be improved by using the WorkStream functionality of dealii
       */
      
      void
      assemble_matrixbased( const VectorType & levelset_old, 
                            SparseMatrixType & matrix, 
                            VectorType & rhs ) const override
      {
        
        levelset_old.update_ghost_values();

        const auto mapping = matrix_free.get_mapping_info().mapping;

        FEValues<dim> fe_values( *mapping,
                                 matrix_free.get_dof_handler().get_fe(),
                                 matrix_free.get_quadrature(),
                                 update_values | update_gradients | update_quadrature_points | update_JxW_values
                                 );
        const unsigned int                    dofs_per_cell =matrix_free.get_dofs_per_cell();
        
        FullMatrix<double>   cell_matrix( dofs_per_cell, dofs_per_cell );
        Vector<double>       cell_rhs(    dofs_per_cell );
        
        const unsigned int n_q_points = fe_values.get_quadrature().size();

        std::vector<double>         psi_at_q(      n_q_points );
        std::vector<Tensor<1,dim>>  grad_psi_at_q( n_q_points, Tensor<1,dim>() );
        std::vector<Tensor<1,dim>>  normal_at_q(   n_q_points, Tensor<1,dim>() );
        
        std::vector<types::global_dof_index> local_dof_indices( dofs_per_cell );

        rhs      = 0.0;
        matrix   = 0.0;
        
        this->n.update_ghost_values();

        for (const auto &cell : matrix_free.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
        {
          cell_matrix = 0.0;
          cell_rhs    = 0.0;
          fe_values.reinit(cell);

          const double epsilon_cell = eps>0.0 ? eps : cell->diameter() / ( std::sqrt(dim) * 2 );
          AssertThrow(epsilon_cell>0.0, ExcMessage("Reinitialization: the value of epsilon for the reinitialization function must be larger than zero!"));

          fe_values.get_function_values(     levelset_old, psi_at_q );     // compute values of old solution at tau_n
          fe_values.get_function_gradients(  levelset_old, grad_psi_at_q ); // compute gradients of old solution at tau_n
          NormalVectorNew::NormalVectorOperator<dim,degree>::get_unit_normals_at_quadrature(fe_values,
                                         n,
                                         normal_at_q);

          for (const unsigned int q_index : fe_values.quadrature_point_indices())
          {
            for (const unsigned int i : fe_values.dof_indices())
            {
              const double nTimesGradient_i = normal_at_q[q_index] * fe_values.shape_grad(i, q_index);

              for (const unsigned int j : fe_values.dof_indices())
              {
                  const double nTimesGradient_j = normal_at_q[q_index] * fe_values.shape_grad(j, q_index);
                   //clang-format off
                  cell_matrix(i,j) += (
                                        fe_values.shape_value(i,q_index) * fe_values.shape_value(j,q_index)
                                        + 
                                        this->d_tau * epsilon_cell * nTimesGradient_i * nTimesGradient_j
                                      ) 
                                      * 
                                      fe_values.JxW( q_index );
                   //clang-format on
              }
              
              const double diffRhs = epsilon_cell * normal_at_q[q_index] * grad_psi_at_q[q_index];

               //clang-format off
              const auto compressive_flux = [](const double psi) { return 0.5 * ( 1. - psi * psi ); };
              cell_rhs(i) += ( compressive_flux(psi_at_q[q_index]) - diffRhs )
                              *
                              nTimesGradient_i 
                              *
                              this->d_tau 
                              * 
                              fe_values.JxW( q_index );
               //clang-format on
            }                                    
          } //end loop over gauss points
          // assembly
          cell->get_dof_indices( local_dof_indices );
          
          this->constraints->distribute_local_to_global( cell_matrix,
                                                   cell_rhs,
                                                   local_dof_indices,
                                                   matrix,
                                                   rhs);
           
        }

        matrix.compress( VectorOperation::add );
        rhs.compress(    VectorOperation::add );
        
      }

      /*
       *    matrix-free implementation 
       *      
       */

      void
      vmult(VectorType & dst,
            const VectorType & src) const override
      {
        AssertThrow(this->d_tau>0.0, ExcMessage("reinitialization operator: d_tau must be set"));
        AssertThrow(eps>0.0, ExcMessage("reinitialization operator: epsilon must be set"));
        
        const int n_q_points_1d = degree+1; //@ get out of matrixfree?
        
        FEEvaluation<dim, degree, n_q_points_1d, 1, number>   levelset(      matrix_free );
        FEEvaluation<dim, degree, n_q_points_1d, dim, number> normal_vector( matrix_free );

        matrix_free.template cell_loop<VectorType, VectorType>( [&] 
          (const auto&, auto& dst, const auto& src, auto cell_range) {
            for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
            {
              levelset.reinit(cell);
              levelset.gather_evaluate(src, true, true);
              
              normal_vector.reinit(cell);
              normal_vector.read_dof_values(this->n);
              normal_vector.evaluate(true, false);

              for (unsigned int q_index=0; q_index<levelset.n_q_points; q_index++)
              {
                  const scalar phi =      levelset.get_value(    q_index );
                  
                  const vector grad_phi = levelset.get_gradient( q_index );
                  
                  const auto n_phi = normalize(normal_vector.get_value(q_index));
                  
                  levelset.submit_value(phi, q_index);
                  levelset.submit_gradient(this->d_tau * eps * scalar_product(grad_phi, n_phi) * n_phi, q_index);
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
               const VectorType & src) const override
    {
      AssertThrow(this->d_tau>0.0, ExcMessage("reinitialization matrix-free operator: d_tau must be set"));
      AssertThrow(eps>0.0,         ExcMessage("reinitialization matrix-free operator: epsilon must be set"));
      
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
            normal_vector.read_dof_values(n);
            normal_vector.evaluate(true, false);

            for (unsigned int q_index = 0; q_index < psi.n_q_points; ++q_index)
            {
              const scalar val = psi.get_value(q_index);
              const auto n_phi = normalize(normal_vector.get_value(q_index));
              
              psi.submit_gradient( this->d_tau * compressive_flux(val) * n_phi 
                                   - 
                                   this->d_tau * eps * scalar_product( psi.get_gradient(q_index), n_phi ) * n_phi, q_index);
            }

            psi.integrate_scatter(false, true, dst);
          }
        },
        dst,
        src,
        true);
    }

    private:
      void
      set_normal_vector_field(const BlockVectorType & normal_vector) 
      {
        n.reinit(dim);
        for (unsigned int d=0; d<dim; ++d)
        {
          matrix_free.initialize_dof_vector(n.block(d));
          n.block(d).copy_locally_owned_data_from(normal_vector.block(d));
        }
        n.update_ghost_values();
      }

      
      static
      vector
      normalize(const scalar & in)
      {
          vector vec;
          
          for(unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
            vec[0][v] = in[v] >= 0.0 ? 1.0 : -1.0;
          
          return vec;
      }
      
      static
      vector
      normalize(const vector & in)
      {
          return in / in.norm();
      }

      SmartPointer<const AffineConstraints<number>>     constraints;
      const MatrixFree<dim, double, VectorizedArray<double>>& matrix_free;
      
      double eps = -1.0; 
      BlockVectorType                                 n;
};
}   // namespace Reinitialization
} // namespace MeltPoolDG
