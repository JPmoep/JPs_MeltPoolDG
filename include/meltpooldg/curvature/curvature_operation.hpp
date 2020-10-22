/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, August 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
// MeltPoolDG
#include <meltpooldg/utilities/utilityfunctions.hpp>
#include <meltpooldg/utilities/linearsolve.hpp>
#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/normal_vector/normal_vector_operation.hpp>
#include <meltpooldg/curvature/curvature_operator.hpp>

namespace MeltPoolDG
{
namespace Curvature
{
  using namespace dealii; 

  template <int dim, unsigned int comp=0>
  class CurvatureOperation
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
  private:
    using VectorType          = LinearAlgebra::distributed::Vector<double>;         
    using BlockVectorType     = LinearAlgebra::distributed::BlockVector<double>;    
    using SparseMatrixType    = TrilinosWrappers::SparseMatrix;                     

  public:
    /*
     *  In this struct, the main parameters of the curvature class are stored.
     */
    CurvatureData<double> curvature_data;
    /*
     *    This is the primary solution variable of this module, which will be also publically 
     *    accessible for output_results.
     */
    VectorType solution_curvature;
    
    CurvatureOperation() = default; 

    void
    initialize( const std::shared_ptr<const ScratchData<dim>> & scratch_data_in,
                const Parameters<double>& data_in )
    {
      scratch_data = scratch_data_in;
      /*
       *  initialize curvature data
       */
      curvature_data = data_in.curv;
      /*
       *    initialize normal_vector_operation for computing the normal vector to the given
       *    scalar function for which the curvature should be calculated.
       */
      normal_vector_operation.initialize( scratch_data, data_in );
      /*
       *  initialize the operator (input-dependent: matrix-based or matrix-free)
       */
      create_operator();
    }

    void
    solve( const VectorType& solution_levelset )
    {
      /*
       *    compute and solve the normal vector field for the given level set
       */
      normal_vector_operation.solve( solution_levelset );
      
      VectorType rhs;

      scratch_data->initialize_dof_vector(rhs, comp);
      scratch_data->initialize_dof_vector(solution_curvature, comp);
      int iter = 0;
      
      if (curvature_data.do_matrix_free)
      {
        curvature_operator->create_rhs( rhs, normal_vector_operation.solution_normal_vector);
        iter = LinearSolve< VectorType,
                            SolverCG<VectorType>,
                            OperatorBase<double, VectorType, BlockVectorType>>
                            ::solve( *curvature_operator,
                                     solution_curvature,
                                     rhs );
      }
      else
      {
        curvature_operator->assemble_matrixbased( normal_vector_operation.solution_normal_vector, 
                                                  curvature_operator->system_matrix, 
                                                  rhs );

        iter = LinearSolve<VectorType,
                             SolverCG<VectorType>,
                             SparseMatrixType>::solve( curvature_operator->system_matrix,
                                                       solution_curvature,
                                                       rhs );

      }

      scratch_data->get_constraint(comp).distribute(solution_curvature);

      if (curvature_data.do_print_l2norm)
      {
        const ConditionalOStream& pcout = scratch_data->get_pcout();
        pcout <<  "| curvature:         i=" << iter << " \t"; 
        pcout << "|k| = " << std::setprecision(11) << std::setw(15) << std::left << solution_curvature.l2_norm();
        pcout << std::endl;
      }
    }
  
  private:
    void create_operator()
    {
      const double damping_parameter = scratch_data->get_min_cell_size(comp) * curvature_data.damping_scale_factor;
      curvature_operator = std::make_unique<CurvatureOperator<dim, comp>>(
                                                          *scratch_data,
                                                          damping_parameter );
      /*
       *  In case of a matrix-based simulation, setup the distributed sparsity pattern and
       *  apply it to the system matrix. This functionality is part of the OperatorBase class.
       */
      if (!curvature_data.do_matrix_free)
        curvature_operator->initialize_matrix_based<dim,comp>(*scratch_data);
    }

  private:
    std::shared_ptr<const ScratchData<dim>> scratch_data;
    
    NormalVector::NormalVectorOperation<dim,comp> normal_vector_operation;

    /* 
     *  This pointer will point to your user-defined curvature operator.
     */
    std::unique_ptr<OperatorBase<double, VectorType, BlockVectorType>>    
                                               curvature_operator;    
  };
} // namespace Curvature
} // namespace MeltPoolDG
