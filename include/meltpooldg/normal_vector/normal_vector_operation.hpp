/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, August 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
 //for using smart pointers
#include <deal.II/base/smartpointer.h>

// MeltPoolDG
#include <meltpooldg/utilities/utilityfunctions.hpp>
#include <meltpooldg/utilities/linearsolve.hpp>
#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/normal_vector/normal_vector_operator.hpp>

namespace MeltPoolDG
{
namespace NormalVector
{
  using namespace dealii; 

  /*
   *  This function calculates the normal vector of the current level set function being
   *  the solution of an intermediate projection step 
   *   
   *              (w, n_ϕ)  + η_n (∇w, ∇n_ϕ)  = (w,∇ϕ)
   *                      Ω                 Ω            Ω            
   *  
   *  with test function w, the normal vector n_ϕ, damping parameter η_n and the
   *  level set function ϕ.
   *
   *    !!!! 
   *          the normal vector field is not normalized to length one, 
   *          it actually represents the gradient of the level set 
   *          function 
   *    !!!! 
   */
  
  template <int dim, unsigned int comp=0>
  class NormalVectorOperation
  {
  private:
    using VectorType          = LinearAlgebra::distributed::Vector<double>;         
    using BlockVectorType     = LinearAlgebra::distributed::BlockVector<double>;    
    using SparseMatrixType    = TrilinosWrappers::SparseMatrix;                     

  public:
    NormalVectorData<double> normal_vector_data;
    /*
     *    This is the primary solution variable of this module, which will be also publically 
     *    accessible for output_results.
     */
    BlockVectorType solution_normal_vector;

    NormalVectorOperation() = default;
    
    void
    initialize( const std::shared_ptr<const ScratchData<dim>> & scratch_data_in,
                const Parameters<double>& data_in )
    {
      scratch_data = scratch_data_in;
      /*
       *  initialize normal vector data
       */
      normal_vector_data = data_in.normal_vec;
      /*
       *  initialize normal vector operator
       */
      create_operator();
    }

    void
    solve( const VectorType& solution_levelset_in)
    {
      BlockVectorType rhs;
      
      scratch_data->initialize_block_dof_vector(rhs);
      scratch_data->initialize_block_dof_vector(solution_normal_vector);
      
      int iter = 0;
      
      if (normal_vector_data.do_matrix_free)
      {
        normal_vector_operator->create_rhs( rhs, solution_levelset_in );
        iter = LinearSolve< BlockVectorType,
                            SolverCG<BlockVectorType>,
                            OperatorBase<double, BlockVectorType, VectorType>>
                            ::solve( *normal_vector_operator,
                                     solution_normal_vector,
                                     rhs );
        solution_normal_vector.update_ghost_values();
      }
      else
      {
        
        normal_vector_operator->assemble_matrixbased( solution_levelset_in, 
                                                      normal_vector_operator->system_matrix, 
                                                      rhs );

        for (unsigned int d=0; d<dim; ++d)
        {
          iter = LinearSolve<VectorType,
                             SolverCG<VectorType>,
                             SparseMatrixType>::solve( normal_vector_operator->system_matrix,
                                                       solution_normal_vector.block(d),
                                                       rhs.block(d) );

          scratch_data->get_constraint(comp).distribute(solution_normal_vector.block(d));
          solution_normal_vector.block(d).update_ghost_values();
        }
      }

      if (normal_vector_data.do_print_l2norm)
      {
        const ConditionalOStream& pcout = scratch_data->get_pcout();
        pcout <<  "| normal vector:         i=" << iter << " \t"; 
        for(unsigned int d=0; d<dim; ++d)
          pcout << "|n_" << d << "| = " << std::setprecision(11) << std::setw(15) << std::left << solution_normal_vector.block(d).l2_norm();
        pcout << std::endl;
      }
    }
  
  private:
    void create_operator()
    {
      const double damping_parameter = scratch_data->get_min_cell_size() * normal_vector_data.damping_scale_factor;
      normal_vector_operator = std::make_unique<NormalVectorOperator<dim, comp>>( *scratch_data,
                                                                                           damping_parameter );
      /*
       *  In case of a matrix-based simulation, setup the distributed sparsity pattern and
       *  apply it to the system matrix. This functionality is part of the OperatorBase class.
       */
      if (!normal_vector_data.do_matrix_free)
        normal_vector_operator->initialize_matrix_based<dim,comp>(*scratch_data);
    }
  private:
    std::shared_ptr<const ScratchData<dim>> scratch_data;

    /* 
     *  This pointer will point to your user-defined normal vector operator.
     */
    std::unique_ptr<OperatorBase<double, BlockVectorType, VectorType>>    
                                               normal_vector_operator;    
  };
} // namespace NormalVector
} // namespace MeltPoolDG
