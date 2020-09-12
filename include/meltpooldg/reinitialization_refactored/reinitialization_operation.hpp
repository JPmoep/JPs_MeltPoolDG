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
#include <meltpooldg/normal_vector_refactored/normal_vector_operation.hpp>
#include <meltpooldg/reinitialization_refactored/olsson_operator.hpp>

namespace MeltPoolDG
{
namespace ReinitializationNew
{
  using namespace dealii; 

  /*
   *    Data for reinitialization of level set equation
   */
  
  enum class ReinitModelType { olsson2007 = 1,  //@ todo: number can be removed when input parameter of model type is changed to string
                               /* ... your reinitialization operator ...*/
                               undefined  };
  
  struct ReinitializationData
  {
    // enum which reinitialization model should be solved
    ReinitModelType reinit_model = ReinitModelType::undefined;
    
    // time step for reinitialization
    double d_tau = 0.01;
    
    // choose a constant, not cell-size dependent smoothing parameter
    double constant_epsilon = -1.0;

    // maximum number of reinitialization steps to be completed
    unsigned int max_reinit_steps = 5;
    
    // this parameter controls whether the l2 norm is printed (mainly for testing purposes)
    bool do_print_l2norm = false;
    
    // this parameter activates the matrix free cell loop procedure
    bool do_matrix_free = false;
    
    // maximum number of reinitialization steps to be completed
    TypeDefs::VerbosityType verbosity_level = TypeDefs::VerbosityType::silent;
  };
  
  /*
   *     Reinitialization model for reobtaining the signed-distance 
   *     property of the level set equation
   */
  
  template <int dim, int degree, unsigned int comp=0>
  class ReinitializationOperation 
  {
  private:
    using VectorType          = LinearAlgebra::distributed::Vector<double>;         
    using BlockVectorType     = LinearAlgebra::distributed::BlockVector<double>;    
    using SparseMatrixType    = TrilinosWrappers::SparseMatrix;                     
    using SparsityPatternType = TrilinosWrappers::SparsityPattern;
    using ConstraintsType     = AffineConstraints<double>;   

  public:
    ReinitializationData reinit_data;
    /*
     *    This is the primary solution variable of this module, which will be also publically 
     *    accessible for output_results.
     */
    VectorType           solution_levelset;

    ReinitializationOperation( MatrixFree<dim, double, VectorizedArray<double>>& scratch_data_in,
                               const ConstraintsType&                            constraints_in,
                               const double                                      min_cell_size_in)
      : scratch_data(           scratch_data_in )
      , constraints           ( &constraints_in )
      , min_cell_size         ( min_cell_size_in )
      , mpi_communicator      ( MPI_COMM_WORLD ) // @todo!!!!!!!!!!!!
      //, mpi_communicator      ( UtilityFunctions::get_mpi_comm(scratch_data_in.get_dof_handler(0)) )
      , pcout                 ( std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0 )
      , normal_vector_field   ( scratch_data_in,
                                constraints_in,
                                min_cell_size_in )
    {
    }
    
    void 
    initialize(const VectorType & solution_in,
               const Parameters<double>& data_in )
    {
      scratch_data.initialize_dof_vector(solution_levelset,comp); 
      
      solution_levelset.copy_locally_owned_data_from(solution_in);
      solution_levelset.update_ghost_values();
      /*
       *    initialize the (local) parameters of the reinitialization
       *    from the global user-defined parameters
       */
      set_reinitialization_parameters(data_in);
      /*
       *    initialize normal_vector_field
       */
      normal_vector_field.initialize( data_in );
      /*
       *    update normal vector field
       */
      normal_vector_field.solve( solution_levelset );
      /*
       *   create reinitialization operator. This class supports matrix-based
       *   and matrix-free computation.
       */
      create_operator();
    }

    
    void
    solve()
    {
      VectorType src, rhs;

      scratch_data.initialize_dof_vector(src);
      scratch_data.initialize_dof_vector(rhs);
      
      reinit_operator->set_time_increment(reinit_data.d_tau);

      int iter = 0;
      
      if (reinit_data.do_matrix_free)
      {
        VectorType src_rhs;
        scratch_data.initialize_dof_vector(src_rhs);
        src_rhs.copy_locally_owned_data_from(solution_levelset);
        src_rhs.update_ghost_values();
        reinit_operator->create_rhs( rhs, src_rhs);
        iter = LinearSolve< VectorType,
                            SolverCG<VectorType>,
                            OperatorBase<double>>
                            ::solve( *reinit_operator,
                                      src,
                                      rhs );
      }
      else
      {
        system_matrix.reinit( dsp );  

        TrilinosWrappers::PreconditionAMG preconditioner;     
        TrilinosWrappers::PreconditionAMG::AdditionalData data;     

        preconditioner.initialize(system_matrix, data); 
        reinit_operator->assemble_matrixbased( solution_levelset, system_matrix, rhs );
        iter = LinearSolve<VectorType,
                                     SolverCG<VectorType>,
                                     SparseMatrixType,
                                     TrilinosWrappers::PreconditionAMG>::solve( system_matrix,
                                                                                src,
                                                                                rhs,
                                                                                preconditioner);
        constraints->distribute(src);
      }

      solution_levelset += src;
      
      solution_levelset.update_ghost_values();

      if(reinit_data.do_print_l2norm)
      {
        pcout << "| CG: i=" << std::setw(5) << std::left << iter;
        pcout << "\t |ΔΨ|∞ = " << std::setw(15) << std::left << std::setprecision(10) << src.linfty_norm();
        pcout << " |ΔΨ|²/dT = " << std::setw(15) << std::left << std::setprecision(10) << src.l2_norm()/reinit_data.d_tau << "|" << std::endl;
      }
  }

  private:
    void 
    set_reinitialization_parameters(const Parameters<double>& data_in)
    {
        //@ todo: add parameter for paraview output
      reinit_data.reinit_model        = static_cast<ReinitModelType>(data_in.reinit_modeltype);
      reinit_data.d_tau               = data_in.reinit_dtau > 0.0 ? 
                                        data_in.reinit_dtau
                                        : min_cell_size;
      reinit_data.constant_epsilon    = data_in.reinit_constant_epsilon;
      reinit_data.max_reinit_steps    = data_in.reinit_max_n_steps; //parameters.max_reinitializationsteps;
      //reinit_data.verbosity_level     = TypeDefs::VerbosityType::major;
      reinit_data.do_print_l2norm     = data_in.reinit_do_print_l2norm; //parameters.output_norm_levelset;
      reinit_data.do_matrix_free      = data_in.reinit_do_matrixfree;  
    }

    
    void create_operator()
    {
      if (!reinit_data.do_matrix_free)
      {

        IndexSet locally_owned_dofs;
        IndexSet locally_relevant_dofs;
        
        locally_owned_dofs = scratch_data.get_dof_handler(comp).locally_owned_dofs(); 
        DoFTools::extract_locally_relevant_dofs(scratch_data.get_dof_handler(comp), locally_relevant_dofs);
        
        dsp.reinit( locally_owned_dofs,
                    locally_owned_dofs,
                    locally_relevant_dofs,
                    mpi_communicator);
        DoFTools::make_sparsity_pattern(scratch_data.get_dof_handler(comp), 
                                        dsp,
                                        *constraints,
                                        true,
                                        Utilities::MPI::this_mpi_process(mpi_communicator)
                                        );
        dsp.compress();
        
        system_matrix.reinit( dsp );  
      }
      
      if (reinit_data.reinit_model == ReinitModelType::olsson2007)
      {
       reinit_operator = 
          std::make_unique<OlssonOperator<dim, degree, double>>( reinit_data.d_tau,
                                                                 normal_vector_field.solution_normal_vector, 
                                                                 scratch_data,
                                                                 constraints,
                                                                 min_cell_size
                                                                );
      }
      /* 
       * add your desired operators
       *
       * else if (reinit_data.reinitmodel == ReinitModelType::my_new_reinitialization_model
       *    ....
       */
      else
        AssertThrow(false, ExcMessage("Requested reinitialization model not implemented."))
    }
  
  private:
    MatrixFree<dim, double, VectorizedArray<double>>&       scratch_data;
    SmartPointer<const ConstraintsType>                     constraints;
    double                                                  min_cell_size;           
    const MPI_Comm                                          mpi_communicator;
    ConditionalOStream                                      pcout;                   // @todo: reference
    /*
     *  This shared pointer will point to your user-defined reinitialization operator.
     */
    std::unique_ptr<OperatorBase<double>>      reinit_operator;
    
    /*
    * the following are prototypes for matrix-based operators
    */
    SparsityPatternType                        dsp;
    SparseMatrixType                           system_matrix;     // @todo: might not be a member variable
  public:
    /*
     *   Computation of the normal vectors
     */
    NormalVectorNew::NormalVectorOperation<dim,degree> normal_vector_field;
    
  };
} // namespace ReinitializationNew
} // namespace MeltPoolDG
