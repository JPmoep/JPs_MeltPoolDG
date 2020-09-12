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
#include <meltpooldg/advection_diffusion/advection_diffusion_operation.hpp>

namespace MeltPoolDG
{
namespace AdvectionDiffusion
{
using namespace dealii;

struct AdvectionDiffusionData 
{
  // time step for AdvectionDiffusion
  double dt = 0.01;
  
  // choose the diffusivity parameter
  double diffusivity = 0.0;
  
  // choose theta from the generaliezd time-stepping included
  double theta = 0.5;

  // this parameter controls whether the l2 norm is printed (mainly for testing purposes)
  bool do_print_l2norm = false;
  
  // this parameter activates the matrix free cell loop procedure
  bool do_matrix_free = false;
  
  // maximum number of AdvectionDiffusion steps to be completed
  TypeDefs::VerbosityType verbosity_level = TypeDefs::VerbosityType::silent;
};
  
template<int dim, int degree, typename number = double>
class AdvectionDiffusionOperator : public OperatorBase<number, 
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
    AdvectionDiffusionOperator( const MatrixFree<dim, double, VectorizedArray<double>>& scratch_data_in, 
      SmartPointer<const AffineConstraints<number>>              constraints_in,
      const TensorFunction<1,dim>&                               advection_velocity_in,
      const AdvectionDiffusionData& data_in )
    : matrix_free ( scratch_data_in )
    , constraints         ( constraints_in        )
    , advection_velocity  ( advection_velocity_in )
    , data                ( data_in               )
    {
      this->set_time_increment(data.dt);
    }

    /*
     *    this is the matrix-based implementation of the rhs and the matrix
     *    @todo: this could be improved by using the WorkStream functionality of dealii
     */
    
    void
    assemble_matrixbased( const VectorType & advected_field_old, 
                          SparseMatrixType & matrix, 
                          VectorType & rhs ) const override
    {
      AssertThrow(data.diffusivity>=0.0, ExcMessage("Advection diffusion operator: diffusivity is smaller than zero!"));

      advected_field_old.update_ghost_values();
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

      std::vector<double>                  phi_at_q(      n_q_points );
      std::vector<Tensor<1,dim>>           grad_phi_at_q( n_q_points, Tensor<1,dim>() );
      std::vector<types::global_dof_index> local_dof_indices( dofs_per_cell );

      rhs      = 0.0;
      matrix   = 0.0;
      
      // advection velocity
      Tensor<1, dim> a;
      
      for (const auto &cell : matrix_free.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
      {
        cell_matrix = 0;
        cell_rhs    = 0;
  
        fe_values.reinit(cell);
        fe_values.get_function_values(     advected_field_old, phi_at_q ); 
        fe_values.get_function_gradients(  advected_field_old, grad_phi_at_q ); 
  
        for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          auto qCoord = fe_values.get_quadrature_points()[q_index];
          a =  advection_velocity.value( qCoord );
  
          for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
            {
                auto velocity_grad_phi_j = a * fe_values.shape_grad( j, q_index);  
                // clang-format off
                cell_matrix( i, j ) += ( fe_values.shape_value( i, q_index) 
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
                     data.diffusivity 
                     *
                     fe_values.shape_grad( i, q_index) 
                     *
                     grad_phi_at_q[q_index]
                     +
                     a * grad_phi_at_q[q_index]    
                       * fe_values.shape_value(  i, q_index)
                   )
                ) * fe_values.JxW(q_index) ;      
            // clang-format on
          }
        } // end gauss
    
        // assembly
        cell->get_dof_indices(local_dof_indices);
        constraints->distribute_local_to_global(cell_matrix,
                                               cell_rhs,
                                               local_dof_indices,
                                               matrix,
                                               rhs);
  
      }

      matrix.compress( VectorOperation::add );
      rhs.compress(    VectorOperation::add );
    }

    /*
     *    matrix-free implementation  --> @todo -- wip!!
     *      
     */

    //void
    //vmult(VectorType & dst,
          //const VectorType & src) const override
    //{
    //}


    //void
    //create_rhs(VectorType & dst,
               //const VectorType & src) const override
    //{
    //}

    void
    initialize_dof_vector(VectorType &dst) const
    {
      matrix_free.initialize_dof_vector(dst);
    }

    private:
      const MatrixFree<dim, double, VectorizedArray<double>>& matrix_free;
      SmartPointer<const AffineConstraints<number>>           constraints;
      const TensorFunction<1,dim>&                            advection_velocity;
      const AdvectionDiffusionData&                           data;
};
}   // namespace AdvectionDiffusion
} // namespace MeltPoolDG
