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
#include "meltpooldg/reinitialization/olsson_operator.hpp"

using namespace dealii;

namespace MeltPoolDG
{
namespace Curvature
{
    /*
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
  template<int dim, unsigned int comp=0, typename number = double>
  class CurvatureOperator: public OperatorBase<number, 
                                LinearAlgebra::distributed::Vector<number>, 
                                LinearAlgebra::distributed::BlockVector<number>>
  {
    public:
      using VectorType          = LinearAlgebra::distributed::Vector<number>;
      using BlockVectorType     = LinearAlgebra::distributed::BlockVector<number>;
      using VectorizedArrayType = VectorizedArray<number>;
      using SparseMatrixType    = TrilinosWrappers::SparseMatrix;                     
      // clang-format off
      CurvatureOperator( const ScratchData<dim>& scratch_data_in,
                         const double            damping_in )
      : scratch_data( scratch_data_in )
      , damping     ( damping_in      )
      {
      }
      // clang-format on

      void
      assemble_matrixbased( const BlockVectorType & solution_normal_vector_in, 
                            SparseMatrixType & matrix, 
                            VectorType & rhs ) const override
      {
      const auto& mapping = scratch_data.get_mapping();     
      FEValues<dim> fe_values( mapping,
                               scratch_data.get_matrix_free().get_dof_handler().get_fe(),
                               scratch_data.get_matrix_free().get_quadrature(),
                               update_values | update_gradients | update_quadrature_points | update_JxW_values );

      const unsigned int                    dofs_per_cell =scratch_data.get_n_dofs_per_cell();

      FullMatrix<double>                    curvature_cell_matrix( dofs_per_cell, dofs_per_cell );
      Vector<double>           curvature_cell_rhs(dofs_per_cell) ;
      std::vector<types::global_dof_index>  local_dof_indices(  dofs_per_cell );
      
      const unsigned int n_q_points = fe_values.get_quadrature().size();

      std::vector<Tensor<1,dim>>            normal_at_q(  n_q_points, Tensor<1,dim>() );
      
      matrix = 0.0;
      rhs = 0.0;

      for (const auto &cell : scratch_data.get_matrix_free().get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        cell->get_dof_indices( local_dof_indices );
        
        curvature_cell_matrix = 0.0;
        curvature_cell_rhs    = 0.0;
              
        NormalVector::NormalVectorOperator<dim>::get_unit_normals_at_quadrature( fe_values,
                                        solution_normal_vector_in,
                                        normal_at_q );
                 
        for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                const double phi_i             = fe_values.shape_value( i, q_index );
                const Tensor<1,dim> grad_phi_i = fe_values.shape_grad(  i, q_index );
                
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                    const double phi_j             = fe_values.shape_value( j, q_index);
                    const Tensor<1,dim> grad_phi_j = fe_values.shape_grad(  j, q_index);

                    curvature_cell_matrix( i, j ) += ( phi_i * phi_j 
                                                       + 
                                                       damping * grad_phi_i * grad_phi_j  
                                                     )
                                                     * 
                                                     fe_values.JxW( q_index ) ;
                }
                curvature_cell_rhs(i) += ( grad_phi_i
                                         * 
                                         normal_at_q[ q_index ] 
                                         * 
                                         fe_values.JxW( q_index ) );
            }
        }
        
         //assembly
        cell->get_dof_indices(local_dof_indices);
        scratch_data.get_constraint(comp).distribute_local_to_global( curvature_cell_matrix,
                                                 curvature_cell_rhs,
                                                 local_dof_indices,
                                                 matrix,
                                                 rhs);
         
      } // end of cell loop
      matrix.compress( VectorOperation::add );
      rhs.compress(    VectorOperation::add );
    }

    /*
     *  matrix-free utility
     */

    //void
    //vmult(VectorType & dst,
          //const VectorType & src) const override
    //{
      ////const int n_q_points_1d = degree+1; // @ todo: not hard code
      ////FEEvaluation<dim, degree, n_q_points_1d, dim, number>   normal( matrix_free );

      ////matrix_free.template cell_loop<BlockVectorType, BlockVectorType>( [&] 
        ////(const auto&, auto& dst, const auto& src, auto cell_range) {
          ////for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          ////{
            ////normal.reinit(cell);
          /*
           * @ bug? --> the following call yield a compilation error
           */
            //////normal.gather_evaluate(src, true, true);
          ////[> current work around <]
            ////normal.read_dof_values(src);
            ////normal.evaluate(true,true,false);

            ////for (unsigned int q_index=0; q_index<normal.n_q_points; ++q_index)
            ////{
                ////normal.submit_value(              normal.get_value(    q_index ), q_index);
                ////normal.submit_gradient( damping * normal.get_gradient( q_index ), q_index );
              ////}
              /*
              * @ bug? --> the following call yield a compilation error
              */
              //////normal_comp.integrate_scatter(true, true, dst);
          ////[> current work around <]
              ////normal.integrate(true, true);
              ////normal.distribute_local_to_global(dst);
            ////}
          ////},
          ////dst, 
          ////src, 
          ////true );
    //}

    //void
    //create_rhs(VectorType & dst,
               //const BlockVectorType & src) const override
    //{
      ////const int n_q_points_1d = degree+1;

      ////FEEvaluation<dim, degree, n_q_points_1d, dim, number>   normal_vector( matrix_free );
      ////FEEvaluation<dim, degree, n_q_points_1d, 1, number>     levelset( matrix_free );

      ////matrix_free.template cell_loop<BlockVectorType, VectorType>(
        ////[&](const auto &, auto &dst, const auto &src, auto macro_cells) {
          ////for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          ////{
            ////normal_vector.reinit(cell);
            ////levelset.reinit(cell);

            ////levelset.gather_evaluate(src, false, true);
            ////for (unsigned int q_index = 0; q_index < normal_vector.n_q_points; ++q_index)
              ////normal_vector.submit_value( levelset.get_gradient(q_index), q_index );

            ////normal_vector.integrate(true, false);
            ////normal_vector.distribute_local_to_global(dst);
            /*
             * @ bug? --> the following call yield a compilation error
             */
            //////normal_vector.integrate_scatter(true, false, dst);
          ////}
        ////},
        ////dst,
        ////src,
        ////true);
    //}

    private:
      const ScratchData<dim>& scratch_data;

      double damping; 
  };
}   // Curvature
} // MeltPoolDG
