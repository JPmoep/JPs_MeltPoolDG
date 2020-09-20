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
#include <meltpooldg/normal_vector/normal_vector_operation.hpp>
#include <meltpooldg/curvature/curvature_operator.hpp>

namespace MeltPoolDG
{
namespace Curvature
{
  using namespace dealii; 

  /*
   *    Data for computing the curvature of a given scalar field
   *    considering diffusive damping;
   */
  
  struct CurvatureData
  {
    // parameter for diffusive term in computation of normals
    double damping_parameter = 1e-6;
    
    // this parameter controls whether the l2 norm is printed (mainly for testing purposes)
    bool do_print_l2norm = true;
    
    // this parameter controls whether the matrixfree operator is called
    bool do_matrix_free = false;
    
    // current verbosity level --> see possible options in UtilityFunctions
    TypeDefs::VerbosityType verbosity_level = TypeDefs::VerbosityType::silent;

  };
  
  /*
   *     Model for computing the normal vector to a scalar function as a smooth function
   *     @ todo: add equation 
   */
  
  template <int dim, int degree, unsigned int comp=0>
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
    using SparsityPatternType = TrilinosWrappers::SparsityPattern;

  public:
    CurvatureData curvature_data;
    /*
     *    This is the primary solution variable of this module, which will be also publically 
     *    accessible for output_results.
     */
    VectorType solution_curvature;
    
    CurvatureOperation()
    {
    }

    void
    initialize( const std::shared_ptr<const ScratchData<dim>> & scratch_data_in,
                const Parameters<double>& data_in )
    {
      scratch_data = scratch_data_in;
      /*
       *    initialize normal_vector_field
       */
      normal_vector_field.initialize( scratch_data, data_in );
      /*
       *  initialize operator
       */
      create_operator();
    }

    void
    solve( const VectorType& solution_levelset )
    {
      /*
       *    update normal vector field
       */
      normal_vector_field.solve( solution_levelset );
      
      VectorType rhs;

      scratch_data->initialize_dof_vector(rhs);
      scratch_data->initialize_dof_vector(solution_curvature);
      
      int iter = 0;
      
      if (curvature_data.do_matrix_free)
      {
        AssertThrow(false,ExcMessage("curvature matrix-free not yet implemented"));
        //curvature_operator->create_rhs( rhs, normal_vector_field.solution_normal_vector );
        //iter = LinearSolve< BlockVectorType,
                            //SolverCG<BlockVectorType>,
                            //OperatorBase<double, VectorType, BlockVectorType>>
                            //::solve( *curvature_operator,
                                     //solution_curvature,
                                     //rhs );
        //solution_curvature.update_ghost_values();
      }
      else
      {
        
        curvature_operator->assemble_matrixbased( normal_vector_field.solution_normal_vector, 
                                                      system_matrix, 
                                                      rhs );

        iter = LinearSolve<VectorType,
                             SolverCG<VectorType>,
                             SparseMatrixType>::solve( system_matrix,
                                                       solution_curvature,
                                                       rhs );

        scratch_data->get_constraint(comp).distribute(solution_curvature);
        solution_curvature.update_ghost_values();
      }

      if (curvature_data.do_print_l2norm)
      {
        const ConditionalOStream& pcout = scratch_data->get_pcout();
        pcout <<  "| curvature:         i=" << iter << " \t"; 
        pcout << "|k| = " << std::setprecision(11) << std::setw(15) << std::left << solution_curvature.l2_norm();
        pcout << std::endl;
      }
    }

    void 
    print_me(); 
  
  private:
    void create_operator()
    {
      if (!curvature_data.do_matrix_free)
      {
        const MPI_Comm mpi_communicator = scratch_data->get_mpi_comm(comp);
        IndexSet locally_owned_dofs;
        IndexSet locally_relevant_dofs;
        
        locally_owned_dofs = scratch_data->get_dof_handler(comp).locally_owned_dofs(); 
        DoFTools::extract_locally_relevant_dofs(scratch_data->get_dof_handler(comp), locally_relevant_dofs);

        dsp.reinit( locally_owned_dofs,
                    locally_owned_dofs,
                    locally_relevant_dofs,
                    mpi_communicator);
        DoFTools::make_sparsity_pattern(scratch_data->get_dof_handler(comp), 
                                        dsp,
                                        scratch_data->get_constraint(comp),
                                        true,
                                        Utilities::MPI::this_mpi_process(mpi_communicator)
                                        );
        dsp.compress();
        
        system_matrix.reinit( dsp );  
      }

      curvature_operator = std::make_unique<CurvatureOperator<dim, degree, comp>>(
                                                          *scratch_data,
                                                          curvature_data.damping_parameter );
    }
    
    std::shared_ptr<const ScratchData<dim>> scratch_data;
    
    NormalVector::NormalVectorOperation<dim,degree> normal_vector_field;

    /* 
     *  This pointer will point to your user-defined curvature operator.
     */
    std::unique_ptr<OperatorBase<double, VectorType, BlockVectorType>>    
                                               curvature_operator;    
    
    SparseMatrixType                           system_matrix;
    SparsityPatternType                        dsp;
  };
} // namespace Curvature
} // namespace MeltPoolDG
