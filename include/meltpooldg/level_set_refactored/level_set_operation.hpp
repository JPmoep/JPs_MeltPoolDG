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
#include <meltpooldg/advection_diffusion/advection_diffusion_operation.hpp>
#include <meltpooldg/reinitialization_refactored/reinitialization_operation.hpp>
#include <meltpooldg/curvature/curvature_operation.hpp>

namespace MeltPoolDG
{
namespace LevelSet
{
  using namespace dealii; 
  using namespace ReinitializationNew; 
  using namespace AdvectionDiffusion; 

  struct LevelSetData 
  {
    // this parameter activates the reinitialization of the level set field
    bool do_reinitialization = false;
    
    // time step for LevelSet
    double dt = 0.01;
    
    // choose the diffusivity parameter
    double artificial_diffusivity = 0.0;
    
    // choose theta from the generaliezd time-stepping included
    double theta = 0.5;

    // this parameter controls whether the l2 norm is printed (mainly for testing purposes)
    bool do_print_l2norm = false;
    
    // maximum number of LevelSet steps to be completed
    TypeDefs::VerbosityType verbosity_level = TypeDefs::VerbosityType::silent;
  };
  
  /*
   *     LevelSet model 
   */
  template <int dim, int degree, int comp=0>
  class LevelSetOperation 
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
    LevelSetData level_set_data;
    /*
     *    This is the primary solution variable of this module, which will be also publically 
     *    accessible for output_results.
     */
    VectorType solution_level_set;
    VectorType& solution_curvature = curvature_operation.solution_curvature;

    LevelSetOperation( MatrixFree<dim, double, VectorizedArray<double>>& scratch_data_in,
                       const ConstraintsType&                            constraints_dir_in,
                       const ConstraintsType&                            constraints_in,
                       TensorFunction<1,dim>&                            advection_velocity_in,
                       const double                                      min_cell_size_in)
      : scratch_data          ( scratch_data_in )
      , min_cell_size         ( min_cell_size_in )
      , mpi_communicator      ( MPI_COMM_WORLD ) 
      //, mpi_communicator      ( UtilityFunctions::get_mpi_comm(*dof_handler) )
      , pcout                 ( std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      , advection_velocity    ( advection_velocity_in )
      , advec_diff_operation(  scratch_data_in,
                           constraints_dir_in,
                           min_cell_size,
                           advection_velocity )
      , reinit_operation(  scratch_data_in,
                           constraints_in,
                           min_cell_size )
      , curvature_operation( scratch_data_in,
                             constraints_in,
                             min_cell_size )
    {
    }
    
    void 
    initialize(const VectorType & solution_in,
               const Parameters<double>& data_in )
    {
      /*
       *  set initiali conditions
       */
      scratch_data.initialize_dof_vector(solution_level_set,comp);

      solution_level_set.copy_locally_owned_data_from(solution_in);
      solution_level_set.update_ghost_values();

      /*
       *  initialize the advection_diffusion problem
       */
      advec_diff_operation.initialize(solution_level_set, 
                                      data_in);
      /*
       *  set the parameters for the levelset problem; already determined parameters
       *  from the initialize call are overwritten.
       */
      set_parameters(data_in);
    }

    
    void
    solve(const Parameters<double>& data_in ) // data_in is needed for reinitialization
    {
      
      /*
       *  solve the advection step of the levelset 
       *    
       */
      advec_diff_operation.solve();
      solution_level_set = advec_diff_operation.solution_advected_field; // @ could be defined by reference

      if(level_set_data.do_reinitialization)
      {
      /*
       *  solve the reinitialization step
       *  
       */
        reinit_operation.initialize(solution_level_set, 
                                    data_in);

        while ( !reinit_time_iterator.is_finished() )
        {
          pcout << std::setw(4) << "" << "| reini: τ= " << std::setw(10) << std::left << reinit_time_iterator.get_current_time();
          reinit_operation.reinit_data.d_tau = reinit_time_iterator.get_next_time_increment();   
          reinit_operation.solve();
        }
        reinit_time_iterator.reset();
      }
      /*
       *    initialize the curvature operation class
       */
      curvature_operation.initialize(solution_level_set, data_in);
      curvature_operation.solve();
    }  
  

  private:
    // @ todo: migrate this function to data struct
    void 
    set_parameters(const Parameters<double>& data_in)
    {
      level_set_data.do_reinitialization = data_in.ls_do_reinitialization;
        //@ todo: add parameter for paraview output
      advec_diff_operation.advec_diff_data.dt               = data_in.ls_time_step_size > 0.0 ? 
                                         data_in.ls_time_step_size
                                        : min_cell_size;
      advec_diff_operation.advec_diff_data.diffusivity      = data_in.ls_artificial_diffusivity;
      advec_diff_operation.advec_diff_data.do_print_l2norm  = true; 
      advec_diff_operation.advec_diff_data.do_matrix_free   = false; // @ todo  
      /*
       *  setup the time iterator for the reinitialization
       */
      reinit_time_iterator.initialize(TimeIteratorData{0.0,
                                                       100000.,
                                                       data_in.reinit_dtau > 0.0 ? data_in.reinit_dtau : min_cell_size,
                                                       data_in.reinit_max_n_steps,
                                                       false});

    }

    MatrixFree<dim, double, VectorizedArray<double>>&  scratch_data;
    double                                             min_cell_size;           
    const MPI_Comm                                     mpi_communicator;
    ConditionalOStream                                 pcout;                   
    
    const TensorFunction<1,dim> &                      advection_velocity;
     /*
     *  This pointer will point to your user-defined advection_diffusion operator.
     */
    AdvectionDiffusionOperation<dim, degree>           advec_diff_operation;
    ReinitializationOperation<dim, degree>             reinit_operation;
    CurvatureNew::CurvatureOperation<dim, degree>         curvature_operation;

    TimeIterator                                       reinit_time_iterator;
    /*
    * the following are prototypes for matrix-based operators
    */
    SparsityPatternType                                dsp;
    SparseMatrixType                                   system_matrix;     // @todo: might not be a member variable
  };
} // namespace AdvectionDiffusion
} // namespace MeltPoolDG
