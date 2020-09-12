/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
 //for using smart pointers
#include <deal.II/base/smartpointer.h>
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
  
  /*
   *     AdvectionDiffusion model 
   */
  template <int dim, int degree, int comp=0>
  class AdvectionDiffusionOperation 
  {
  private:
    using VectorType          = LinearAlgebra::distributed::Vector<double>;         
    using BlockVectorType     = LinearAlgebra::distributed::BlockVector<double>;    
    using SparseMatrixType    = TrilinosWrappers::SparseMatrix;                     
    using DoFHandlerType      = DoFHandler<dim>;                                    
    using SparsityPatternType = TrilinosWrappers::SparsityPattern;
    using ConstraintsType     = AffineConstraints<double>;   

  public:
    /*
     *  All the necessary parameters are stored in this vector.
     */
    AdvectionDiffusionData advec_diff_data;
    /*
     *    This is the primary solution variable of this module, which will be also publically 
     *    accessible for output_results.
     */
    VectorType solution_advected_field;
    AdvectionDiffusionOperation( MatrixFree<dim, double, VectorizedArray<double>>& scratch_data_in,
                                 const ConstraintsType&      constraints_in,
                                 const double                min_cell_size_in,
                                 const TensorFunction<1,dim> & advection_velocity_in )
      : scratch_data          ( scratch_data_in )
      , constraints           ( &constraints_in )
      , min_cell_size         ( min_cell_size_in )
      , mpi_communicator      ( MPI_COMM_WORLD )
      , pcout                 ( std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      , advection_velocity    ( advection_velocity_in )
    {
    }
    
    void 
    initialize(const VectorType & solution_in,
               const Parameters<double>& data_in )
    {
      scratch_data.initialize_dof_vector(solution_advected_field,comp);

      solution_advected_field.copy_locally_owned_data_from(solution_in);
      solution_advected_field.update_ghost_values();

      /*
       *  set the parameters for the advection_diffusion problem
       */
      set_advection_diffusion_parameters(data_in);
      
      create_operator();
    }

    
    void
    solve()
    {
      VectorType src, rhs;

      advec_diff_operator->initialize_dof_vector(src);
      advec_diff_operator->initialize_dof_vector(rhs);
      
      advec_diff_operator->set_time_increment(advec_diff_data.dt);

      int iter = 0;
      
      if (advec_diff_data.do_matrix_free)
      {
        AssertThrow(false, ExcMessage("not yet implemented! "))
        //advec_diff_operator->create_rhs( rhs, solution_advected_field );
        //iter = LinearSolve< VectorType,
                                      //SolverCG<VectorType>,
                                      //OperatorBase<double>>
                                      //::solve( *advec_diff_operator,
                                                //src,
                                                //rhs );
      }
      else
      {
        
        system_matrix.reinit( dsp );  
        //@todo: which preconditioner?
        //TrilinosWrappers::PreconditionAMG preconditioner;     
        //TrilinosWrappers::PreconditionAMG::AdditionalData data;     

        //preconditioner.initialize(system_matrix, data); 
        advec_diff_operator->assemble_matrixbased( solution_advected_field, system_matrix, rhs );
        iter = LinearSolve<VectorType,
                           SolverGMRES<VectorType>,
                           SparseMatrixType>::solve( system_matrix,
                                                               src,
                                                               rhs );
        constraints->distribute(src);
        solution_advected_field = src;
        solution_advected_field.update_ghost_values();
      }

      if(advec_diff_data.do_print_l2norm)
      {
        pcout << "| GMRES: i=" << std::setw(5) << std::left << iter;
        pcout << "\t |Δϕ|2 = " << std::setw(15) << std::left << std::setprecision(10) << src.l2_norm() << std::endl;
      }
  }

  private:
    // @ todo: migrate this function to data struct
    void 
    set_advection_diffusion_parameters(const Parameters<double>& data_in)
    {
        //@ todo: add parameter for paraview output
      advec_diff_data.dt               = data_in.advec_diff_time_step_size > 0.0 ? 
                                         data_in.advec_diff_time_step_size
                                        : min_cell_size;
      advec_diff_data.diffusivity      = data_in.advec_diff_diffusivity;
      advec_diff_data.do_print_l2norm  = data_in.advec_diff_do_print_l2norm; //parameters.output_norm_levelset;
      advec_diff_data.do_matrix_free   = data_in.advec_diff_do_matrixfree;  
    }

    void create_operator()
    {
      if (!advec_diff_data.do_matrix_free)
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
      
      advec_diff_operator = 
         std::make_unique<AdvectionDiffusionOperator<dim, degree, double>>( scratch_data,
                                                                            constraints,
                                                                            advection_velocity,
                                                                            advec_diff_data
                                                                          );
    }

    MatrixFree<dim, double, VectorizedArray<double>>&       scratch_data;
    SmartPointer<const ConstraintsType>        constraints;
    double                                     min_cell_size;           // @todo: check CFL condition
    const MPI_Comm                             mpi_communicator;
    ConditionalOStream                         pcout;                   // @todo: reference
    
    const TensorFunction<1,dim> &              advection_velocity;
     /*
     *  This pointer will point to your user-defined advection_diffusion operator.
     */
    std::unique_ptr<OperatorBase<double>>      advec_diff_operator;
    
    /*
    * the following are prototypes for matrix-based operators
    */
    SparsityPatternType                       dsp;
    SparseMatrixType                          system_matrix;     // @todo: might not be a member variable
  };
} // namespace AdvectionDiffusion
} // namespace MeltPoolDG
