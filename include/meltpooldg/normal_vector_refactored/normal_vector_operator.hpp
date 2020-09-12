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
#include "meltpooldg/interface/operator_base.hpp"

using namespace dealii;

namespace MeltPoolDG
{
namespace NormalVectorNew
{

  template<int dim, int degree, typename number = double>
  class NormalVectorOperator: public OperatorBase<number, 
                                LinearAlgebra::distributed::BlockVector<number>, 
                                LinearAlgebra::distributed::Vector<number>>
  {
    public:
      using VectorType = LinearAlgebra::distributed::Vector<number>;
      using BlockVectorType = LinearAlgebra::distributed::BlockVector<number>;
      using VectorizedArrayType = VectorizedArray<number>;
      using SparseMatrixType    = TrilinosWrappers::SparseMatrix;                     
      using DoFHandlerType    = DoFHandler<dim>;

      
      NormalVectorOperator
      ( MatrixFree<dim, double, VectorizedArray<double>>& scratch_data_in,
        SmartPointer<const AffineConstraints<number>>     constraints_in,
        const double                                      damping_in
      )
      : matrix_free ( scratch_data_in )
      , constraints ( constraints_in )
      , damping     ( damping_in )
      {
      }

      void
      assemble_matrixbased( const VectorType & levelset_in, 
                            SparseMatrixType & matrix, 
                            BlockVectorType & rhs ) const override
      {
        
      const auto mapping = matrix_free.get_mapping_info().mapping;

      auto q_gauss = QGauss<dim>(degree+1);
      FEValues<dim> fe_values( *mapping,
                               matrix_free.get_dof_handler().get_fe(),
                               matrix_free.get_quadrature(),
                               update_values | update_gradients | update_quadrature_points | update_JxW_values );

      const unsigned int                    dofs_per_cell =matrix_free.get_dofs_per_cell();

      FullMatrix<double>                    normal_cell_matrix( dofs_per_cell, dofs_per_cell );
      std::vector<Vector<double>>           normal_cell_rhs(    dim, Vector<double>(dofs_per_cell) );
      std::vector<types::global_dof_index>  local_dof_indices(  dofs_per_cell );
      
      const unsigned int n_q_points = fe_values.get_quadrature().size();

      std::vector<Tensor<1,dim>>            normal_at_q(  n_q_points, Tensor<1,dim>() );
      
      matrix = 0.0;
      rhs = 0.0;

      for (const auto &cell : matrix_free.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        cell->get_dof_indices( local_dof_indices );
        
        normal_cell_matrix = 0.0;
        for(auto& normal_cell : normal_cell_rhs)
            normal_cell =    0.0;

        fe_values.get_function_gradients( levelset_in, normal_at_q );  //compute normals from level set solution at tau=0
        for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            const double phi_i             = fe_values.shape_value(i, q_index);
            const Tensor<1,dim> grad_phi_i = fe_values.shape_grad(i, q_index);
            
            for (unsigned int j=0; j<dofs_per_cell; ++j)
            {
                const double phi_j             = fe_values.shape_value(j, q_index);
                const Tensor<1,dim> grad_phi_j = fe_values.shape_grad(j, q_index);
                
                 //clang-format off
                normal_cell_matrix( i, j ) += ( 
                                                phi_i * phi_j 
                                                + 
                                                damping * grad_phi_i * grad_phi_j  
                                              )
                                              * 
                                              fe_values.JxW( q_index ) ;
                 //clang-format on
            }
     
            for (unsigned int d=0; d<dim; ++d)
            {
                 //clang-format off
                normal_cell_rhs[d](i) += phi_i
                                         * 
                                         normal_at_q[ q_index ][ d ]  
                                         * 
                                         fe_values.JxW( q_index );
                   //clang-format on
            }
          }
        }
        
         //assembly
        cell->get_dof_indices(local_dof_indices);

        constraints->distribute_local_to_global( normal_cell_matrix,
                                                 local_dof_indices,
                                                 matrix);
        for (unsigned int d=0; d<dim; ++d)
            constraints->distribute_local_to_global( normal_cell_rhs[d],
                                                     local_dof_indices,
                                                     rhs.block(d) );
         
      } // end of cell loop
      matrix.compress( VectorOperation::add );
      rhs.compress(    VectorOperation::add );

    }

    /*
     *  matrix-free utility
     */

    void
    vmult(BlockVectorType & dst,
          const BlockVectorType & src) const override
    {
      const int n_q_points_1d = degree+1; // @ todo: not hard code
      FEEvaluation<dim, degree, n_q_points_1d, dim, number>   normal( matrix_free );

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
               const VectorType & src) const override
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
    initialize_dof_vector(VectorType &dst) const override
    {
      matrix_free.initialize_dof_vector(dst);
    }
    
    void
    initialize_block_dof_vector(BlockVectorType &dst) const override
    {
      dst.reinit(dim);
      for (unsigned int d=0; d<dim; ++d)
        matrix_free.initialize_dof_vector(dst.block(d));
    }

    static
    void
    get_unit_normals_at_quadrature( const FEValues<dim>& fe_values,
                                    const BlockVectorType& normal_vector_field_in, 
                                    std::vector<Tensor<1,dim>>& unit_normal_at_quadrature)
    {
      for (unsigned int d=0; d<dim; ++d )
      {
          std::vector<double> temp ( unit_normal_at_quadrature.size() );
          fe_values.get_function_values(  normal_vector_field_in.block(d), temp); // compute normals from level set solution at tau=0
          for (const unsigned int q_index : fe_values.quadrature_point_indices())
              unit_normal_at_quadrature[ q_index ][ d ] = temp[ q_index ];
      }
      for (auto& n : unit_normal_at_quadrature)
          n /= n.norm(); //@todo: add exception if norm is zero
    }

    private:
      MatrixFree<dim, double, VectorizedArray<double>>& matrix_free;

      SmartPointer<const AffineConstraints<number>>   constraints;

      double damping; 

  };
}   // NormalVectorNew
} // MeltPoolDG
