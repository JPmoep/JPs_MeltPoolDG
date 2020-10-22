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
      solution_normal_vector_in.update_ghost_values();

      const auto& mapping = scratch_data.get_mapping();     
      FEValues<dim> fe_values( mapping,
                               scratch_data.get_matrix_free().get_dof_handler(comp).get_fe(),
                               scratch_data.get_matrix_free().get_quadrature(comp),
                               update_values | update_gradients | update_quadrature_points | update_JxW_values );

      const unsigned int                    dofs_per_cell =scratch_data.get_n_dofs_per_cell(comp);

      FullMatrix<double>                    curvature_cell_matrix( dofs_per_cell, dofs_per_cell );
      Vector<double>                        curvature_cell_rhs(    dofs_per_cell) ;
      std::vector<types::global_dof_index>  local_dof_indices(     dofs_per_cell );
      
      const unsigned int n_q_points = fe_values.get_quadrature().size();

      std::vector<Tensor<1,dim>>            normal_at_q(  n_q_points, Tensor<1,dim>() );
      
      matrix = 0.0;
      rhs = 0.0;

      for (const auto &cell : scratch_data.get_matrix_free().get_dof_handler(comp).active_cell_iterators())
      if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        cell->get_dof_indices( local_dof_indices );
        
        curvature_cell_matrix = 0.0;
        curvature_cell_rhs    = 0.0;
              
        NormalVector::NormalVectorOperator<dim,comp>::get_unit_normals_at_quadrature( fe_values,
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

    void
    vmult(VectorType & dst,
          const VectorType & src) const override
    {

      scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>( [&] 
        (const auto&, auto& dst, const auto& src, auto cell_range) {
          FECellIntegrator<dim, 1, number> curvature( scratch_data.get_matrix_free(), comp, comp);
          for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            curvature.reinit(cell);
            curvature.gather_evaluate(src, true, true);

            for (unsigned int q_index=0; q_index<curvature.n_q_points; ++q_index)
            {
                curvature.submit_value(              curvature.get_value(    q_index ), q_index);
                curvature.submit_gradient( damping * curvature.get_gradient( q_index ), q_index );
            }

            curvature.integrate_scatter(true, true, dst);
          }
        },
        dst, 
        src, 
        true );
    }

    void
    create_rhs(VectorType & dst,
               const BlockVectorType & src) const override
    {
      scratch_data.get_matrix_free().template cell_loop<VectorType, BlockVectorType>(
        [&](const auto &, auto &dst, const auto &src, auto macro_cells) {
          FECellIntegrator<dim, 1, number>   curvature(      scratch_data.get_matrix_free(), comp, comp);
          FECellIntegrator<dim, dim, number> normal_vector(  scratch_data.get_matrix_free(), comp, comp);
          for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
          {
            curvature.reinit(cell);

            normal_vector.reinit(cell);
            normal_vector.read_dof_values_plain(src);
            normal_vector.evaluate(true, false);

            for (unsigned int q_index = 0; q_index < curvature.n_q_points; ++q_index)
            {
              const auto n_phi = Reinitialization::OlssonOperator<dim,comp>::normalize(normal_vector.get_value(q_index));
              curvature.submit_gradient( n_phi, q_index );
            }

            curvature.integrate_scatter(false, true, dst);
          }
        },
        dst,
        src,
        true);
    }

    private:
      const ScratchData<dim>& scratch_data;

      double damping; 
  };
}   // Curvature
} // MeltPoolDG
