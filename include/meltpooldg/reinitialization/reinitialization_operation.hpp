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
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/normal_vector/normal_vector_operation.hpp>
#include <meltpooldg/reinitialization/olsson_operator.hpp>

namespace MeltPoolDG
{
namespace Reinitialization
{
  using namespace dealii; 

  /*
   *     Reinitialization model for reobtaining the signed-distance 
   *     property of the level set equation
   */
  
  template <int dim, unsigned int comp=0>
  class ReinitializationOperation 
  {
  private:
    using VectorType          = LinearAlgebra::distributed::Vector<double>;         
    using BlockVectorType     = LinearAlgebra::distributed::BlockVector<double>;    
    using SparseMatrixType    = TrilinosWrappers::SparseMatrix;                     

  public:
    ReinitializationData<double>      reinit_data;
    /*
     *    This is the primary solution variable of this module, which will be also publically 
     *    accessible for output_results.
     */
    VectorType       solution_level_set;
    const BlockVectorType& solution_normal_vector = normal_vector_operation.solution_normal_vector;

    ReinitializationOperation() = default;
    
    void 
    initialize(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
               const VectorType                              &solution_level_set_in,
               const Parameters<double>&                      data_in )
    {
      scratch_data = scratch_data_in;
      scratch_data->initialize_dof_vector(solution_level_set); 
      /*
       *    initialize the (local) parameters of the reinitialization
       *    from the global user-defined parameters
       */
      set_reinitialization_parameters(data_in);
      /*
       *    initialize normal_vector_field
       */
      normal_vector_operation.initialize( scratch_data_in, data_in );
      /*
       *    compute the normal vector field and update the initial solution
       */
      update_initial_solution(solution_level_set_in);
      /*
       *   create reinitialization operator. This class supports matrix-based
       *   and matrix-free computation.
       */
      create_operator();
    }

    /*
     *  By calling the reinitialize function, (1) the solution_level_set field 
     *  and (2) the normal vector field corresponding to the given solution_level_set_field
     *  is updated. This is commonly the first stage before performing the pesude-time-dependent
     *  solution procedure.
     */
    void
    update_initial_solution(const VectorType & solution_level_set_in)
    {
      /*
       *    copy the given solution into the member variable
       */
      solution_level_set.copy_locally_owned_data_from(solution_level_set_in);
      solution_level_set.update_ghost_values();
      /*
       *    update the normal vector field corresponding to the given solution of the 
       *    level set; the normal vector field is called by reference within the  
       *    operator class
       */
      normal_vector_operation.solve( solution_level_set );
    }

    void
    solve(const double d_tau)
    {
      VectorType src, rhs;

      scratch_data->initialize_dof_vector(src);
      scratch_data->initialize_dof_vector(rhs);
      
      reinit_operator->set_time_increment(d_tau);

      int iter = 0;

      if (reinit_data.do_matrix_free)
      {
        VectorType src_rhs;
        scratch_data->initialize_dof_vector(src_rhs);
        src_rhs.copy_locally_owned_data_from(solution_level_set);
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
        reinit_operator->system_matrix.reinit( reinit_operator->dsp );  

        TrilinosWrappers::PreconditionAMG preconditioner;     
        TrilinosWrappers::PreconditionAMG::AdditionalData data;     

        preconditioner.initialize(reinit_operator->system_matrix, data); 
        reinit_operator->assemble_matrixbased( solution_level_set, reinit_operator->system_matrix, rhs );
        iter = LinearSolve<VectorType,
                                     SolverCG<VectorType>,
                                     SparseMatrixType,
                                     TrilinosWrappers::PreconditionAMG>::solve( reinit_operator->system_matrix,
                                                                                src,
                                                                                rhs,
                                                                                preconditioner);
        scratch_data->get_constraint(comp).distribute(src);
      }

      solution_level_set += src;
      
      solution_level_set.update_ghost_values();
      
      if(reinit_data.do_print_l2norm)
      {
        const ConditionalOStream& pcout = scratch_data->get_pcout(comp);
        pcout << "| CG: i=" << std::setw(5) << std::left << iter;
        pcout << "\t |ΔΨ|∞ = " << std::setw(15) << std::left << std::setprecision(10) << src.linfty_norm();
        pcout << " |ΔΨ|²/dT = " << std::setw(15) << std::left << std::setprecision(10) << src.l2_norm()/d_tau << "|" << std::endl;
      }
  }

  private:
    void 
    set_reinitialization_parameters(const Parameters<double>& data_in)
    {
      reinit_data = data_in.reinit;
    }

    void create_operator()
    {
      
      if (reinit_data.modeltype == "olsson2007")
      {

       reinit_operator = 
          std::make_unique<OlssonOperator<dim, comp, double>>( *scratch_data,
                                                                        normal_vector_operation.solution_normal_vector,
                                                                        reinit_data.constant_epsilon,
                                                                        reinit_data.scale_factor_epsilon
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
      /*
       *  In case of a matrix-based simulation, setup the distributed sparsity pattern and
       *  apply it to the system matrix. This functionality is part of the OperatorBase class.
       */
      if (!reinit_data.do_matrix_free)
        reinit_operator->initialize_matrix_based<dim,comp>(*scratch_data);
    }
  
  private:
    std::shared_ptr<const ScratchData<dim>>                 scratch_data;
    /*
     *  This shared pointer will point to your user-defined reinitialization operator.
     */
    std::unique_ptr<OperatorBase<double>>                   reinit_operator;
    /*
     *   Computation of the normal vectors
     */
    NormalVector::NormalVectorOperation<dim>         normal_vector_operation;
    
  };
} // namespace Reinitialization
} // namespace MeltPoolDG
