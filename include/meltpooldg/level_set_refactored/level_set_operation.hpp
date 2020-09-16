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
    using SparsityPatternType = TrilinosWrappers::SparsityPattern;

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
    //VectorType& solution_curvature = curvature_operation.solution_curvature;

    LevelSetOperation( )
    {
    }
    
    void 
    initialize(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
               const VectorType &                             solution_level_set_in,
               const Parameters<double>&                      data_in,
               const std::shared_ptr<TensorFunction<1,dim>>  &advection_velocity_in )
    {
      /*
       *  set initiali conditions
       */
      scratch_data = scratch_data_in;
      advection_velocity = advection_velocity_in;
      parameters=data_in;

      scratch_data->initialize_dof_vector(solution_level_set,comp);

      solution_level_set.copy_locally_owned_data_from(solution_level_set_in);
      solution_level_set.update_ghost_values();

      /*
       *  initialize the advection_diffusion problem
       */
      advec_diff_operation.initialize(scratch_data,
                                      solution_level_set, 
                                      data_in,
                                      advection_velocity_in);
      /*
       *  set the parameters for the levelset problem; already determined parameters
       *  from the initialize call are overwritten.
       */
      set_level_set_parameters(data_in);
    }

    
    void
    solve(const double dt ) // data_in is needed for reinitialization
    {
      
      /*
       *  solve the advection step of the levelset 
       */
      advec_diff_operation.solve( dt );
      solution_level_set = advec_diff_operation.solution_advected_field; // @ could be defined by reference

      /*
       *  solve the reinitialization of the level set equation
       */
      if(level_set_data.do_reinitialization)
      {
        reinit_operation.initialize(scratch_data,
                                    solution_level_set, 
                                    parameters);

        while ( !reinit_time_iterator.is_finished() )
        {
          const double d_tau = reinit_time_iterator.get_next_time_increment();   
          scratch_data->get_pcout() << std::setw(4) << "" << "| reini: τ= " << std::setw(10) << std::left << reinit_time_iterator.get_current_time();
          reinit_operation.solve(d_tau);
        }
        solution_level_set = reinit_operation.solution_level_set; // @ could be defined by reference
        reinit_time_iterator.reset();
      }
      /*
       *    initialize the curvature operation class
       */
      curvature_operation.initialize(scratch_data, parameters);
      /*
       *    compute the curvature
       */
      curvature_operation.solve(solution_level_set);
    }  
  

  private:
    // @ todo: migrate this function to data struct
    void 
    set_level_set_parameters(const Parameters<double>& data_in)
    {
      level_set_data.do_reinitialization = true; //data_in.ls_do_reinitialization;
        //@ todo: add parameter for paraview output
      advec_diff_operation.advec_diff_data.diffusivity      = data_in.ls_artificial_diffusivity;
      advec_diff_operation.advec_diff_data.do_print_l2norm  = true; 
      advec_diff_operation.advec_diff_data.do_matrix_free   = false; 
      /*
       *  setup the time iterator for the reinitialization
       */
      reinit_time_iterator.initialize(TimeIteratorData{0.0,
                                                       100000.,
                                                       data_in.reinit_dtau > 0.0 ? data_in.reinit_dtau : scratch_data->get_min_cell_size(),
                                                       data_in.reinit_max_n_steps,
                                                       false});

    }

    std::shared_ptr<const ScratchData<dim>>             scratch_data;    
    std::shared_ptr<TensorFunction<1,dim>>              advection_velocity;
    Parameters<double>                                  parameters; //evt. nicht mehr
     /*
     *  This pointer will point to your user-defined advection_diffusion operator.
     */
    AdvectionDiffusionOperation<dim, degree,1>           advec_diff_operation;
    ReinitializationOperation<dim, degree,0>             reinit_operation;
    CurvatureNew::CurvatureOperation<dim, degree, 0>      curvature_operation;

    TimeIterator                                       reinit_time_iterator;
    /*
    * the following are prototypes for matrix-based operators
    */
    SparsityPatternType                                dsp;
    SparseMatrixType                                   system_matrix;     // @todo: might not be a member variable
  };
} // namespace AdvectionDiffusion
} // namespace MeltPoolDG
