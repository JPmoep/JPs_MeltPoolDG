/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
// DoFTools
#include <deal.II/dofs/dof_tools.h>
// MeltPoolDG
#include <meltpooldg/utilities/utilityfunctions.hpp>
#include <meltpooldg/utilities/linearsolve.hpp>
#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/advection_diffusion/advection_diffusion_operator.hpp>

namespace MeltPoolDG
{
namespace AdvectionDiffusion
{
  using namespace dealii; 
  
  template <int dim, int degree, int comp=0>
  class AdvectionDiffusionOperation 
  {
  private:
    using VectorType          = LinearAlgebra::distributed::Vector<double>;         
    using BlockVectorType     = LinearAlgebra::distributed::BlockVector<double>;    
    using SparseMatrixType    = TrilinosWrappers::SparseMatrix;                     
    using SparsityPatternType = TrilinosWrappers::SparsityPattern;

  public:
    /*
     *    This is the primary solution variable of this module, which will be also publically 
     *    accessible for output_results.
     */
    VectorType solution_advected_field;
    /*
     *  All the necessary parameters are stored in this struct.
     */
    AdvectionDiffusionData advec_diff_data;

    AdvectionDiffusionOperation() = default;
    
    void 
    initialize(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
               const VectorType &                             solution_advected_field_in,
               const Parameters<double>&                      data_in,
               const std::shared_ptr<TensorFunction<1,dim>> & advection_velocity_in )
    {
      scratch_data = scratch_data_in;
      /*
       *  set the initial solution of the advected field
       */
      scratch_data->initialize_dof_vector(solution_advected_field,comp);
      solution_advected_field.copy_locally_owned_data_from(solution_advected_field_in);
      solution_advected_field.update_ghost_values();
      /*
       *  copy the given velocity function
       */
      advection_velocity = advection_velocity_in; 
      /*
       *  set the parameters for the advection_diffusion problem
       */
      set_advection_diffusion_parameters(data_in);
      /*
       *  create the advection-diffusion operator
       */
      create_operator();
    }

    
    void
    solve(const double dt)
    {
      VectorType src, rhs;

      scratch_data->initialize_dof_vector(src);
      scratch_data->initialize_dof_vector(rhs);
      
      advec_diff_operator->set_time_increment(dt);
      
      int iter = 0;
      
      if (advec_diff_data.do_matrix_free)
      {
        AssertThrow(false, ExcMessage("not yet implemented! "))
      }
      else
      {
        //@todo: which preconditioner?
        //TrilinosWrappers::PreconditionAMG preconditioner;     
        //TrilinosWrappers::PreconditionAMG::AdditionalData data;     

        //preconditioner.initialize(system_matrix, data); 
        advec_diff_operator->assemble_matrixbased( solution_advected_field, 
                                                   system_matrix, 
                                                   rhs );
        iter = LinearSolve<VectorType,
                           SolverGMRES<VectorType>,
                           SparseMatrixType>::solve( system_matrix,
                                                               src,
                                                               rhs );
        scratch_data->get_constraint(comp).distribute(src);

        solution_advected_field = src;
        solution_advected_field.update_ghost_values();
      }
      
      if(advec_diff_data.do_print_l2norm)
      {
        const ConditionalOStream& pcout = scratch_data->get_pcout();
        pcout << "| GMRES: i=" << std::setw(5) << std::left << iter;
        pcout << "\t |Δϕ|2 = " << std::setw(15) << std::left << std::setprecision(10) << src.l2_norm() << std::endl;
      }
  }

  private:
    // @ todo: migrate this function to parameter class
    void 
    set_advection_diffusion_parameters(const Parameters<double>& data_in)
    {
      advec_diff_data.diffusivity      = data_in.advec_diff_diffusivity;
      advec_diff_data.do_print_l2norm  = data_in.advec_diff_do_print_l2norm; 
      advec_diff_data.do_matrix_free   = data_in.advec_diff_do_matrixfree;  
    }

    void create_operator()
    {
      if (!advec_diff_data.do_matrix_free)
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
      
      advec_diff_operator = 
         std::make_unique<AdvectionDiffusionOperator<dim, degree, comp, double>>( *scratch_data,
                                                                            *advection_velocity,
                                                                            advec_diff_data
                                                                          );
    }

  private:
    std::shared_ptr<const ScratchData<dim>>    scratch_data;    
    std::shared_ptr<TensorFunction<1,dim>>     advection_velocity;
     /*
     *  This pointer will point to your user-defined advection_diffusion operator.
     */
    std::unique_ptr<OperatorBase<double>>      advec_diff_operator;
    
    /*
    * the following are prototypes for matrix-based operators
    */
    SparsityPatternType                       dsp;
    SparseMatrixType                          system_matrix;     
  };
} // namespace AdvectionDiffusion
} // namespace MeltPoolDG
