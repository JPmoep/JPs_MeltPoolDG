/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, August 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>

// enabling conditional ostreams
#include <deal.II/base/conditional_ostream.h> 
// for index set
#include <deal.II/base/index_set.h>
// for mpi
#include <deal.II/base/mpi.h> 
// for quadrature points
#include <deal.II/base/quadrature_lib.h>
// for using smart pointers
#include <deal.II/base/smartpointer.h>
//// for distributed triangulation
//#include <deal.II/distributed/tria.h>
// for dof_handler type
#include <deal.II/dofs/dof_handler.h>
// for FE_Q<dim> type
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
// for FE_Q<dim> type
#include <deal.II/fe/mapping.h>

// from multiphaseflow
#include "utilityFunctions.hpp"
#include "normalvector.hpp"
#include "timeiterator.hpp"

namespace LevelSetParallel
{
  using namespace dealii; 

  /*
   *    Data for reinitialization of level set equation
   */
  
  enum class ReinitModelType {olsson2007, undefined};
  
  struct ReinitializationData
  {
    ReinitializationData()
        : reinit_model(ReinitModelType::undefined)
        , d_tau(0.01)
        , constant_epsilon(0.0)
        , degree(1)
        , max_reinit_steps(5)
        , verbosity_level(utilityFunctions::VerbosityType::silent)
        , min_cell_size(0.0)
        , do_print_l2norm(false)
        , do_matrix_free(false)
    {
    }

    // enum which reinitialization model should be solved
    ReinitModelType reinit_model;
    
    // time step for reinitialization
    double d_tau;
    
    // choose a constant, not cell-size dependent smoothing parameter
    double constant_epsilon;
    
    // interpolation degree of reinitalization function
    unsigned int degree;

    // maximum number of reinitialization steps to be completed
    unsigned int max_reinit_steps;
    
    // maximum number of reinitialization steps to be completed
    utilityFunctions::VerbosityType verbosity_level;
    
    // minimum cell size --> to compute CFL condition
    double min_cell_size;
    
    // this parameter controls whether the l2 norm is printed (mainly for testing purposes)
    bool do_print_l2norm;
    
    // this parameter activates the matrix free cell loop procedure
    bool do_matrix_free;

    // @ add lambda function for calculating epsilon
  };
  
  /*
   *     Reinitialization model for reobtaining the signed-distance 
   *     property of the  level set equation
   */
  
  template <int dim>
  class Reinitialization
  {
  private:
      
    typedef LinearAlgebra::distributed::Vector<double>      VectorType;
    typedef LinearAlgebra::distributed::BlockVector<double> BlockVectorType;
    typedef TrilinosWrappers::SparseMatrix                  SparseMatrixType;

    typedef DoFHandler<dim>                                 DoFHandlerType;
    
    typedef DynamicSparsityPattern                          SparsityPatternType;
    
    typedef AffineConstraints<double>                       ConstraintsType;

  public:

    /*
     *  Constructor
     */
    Reinitialization(const MPI_Comm & mpi_commun_in);

    void
    initialize( const ReinitializationData &     data_in,
                const SparsityPatternType&       dsp_in,
                DoFHandler<dim> const &          dof_handler_in,
                const ConstraintsType&           constraints_in,
                const IndexSet&                  locally_owned_dofs_in,
                const IndexSet&                  locally_relevant_dofs_in);

    /*
     *  This function reinitializes the solution of the level set equation for a given solution
     */
    void 
    solve( VectorType & solution_out );

    void 
    print_me(); 
  
  private:
    /* Olsson, Kreiss, Zahedi (2007) model 
     *
     * for reinitialization of the level set equation 
     * 
     */
    void 
    solve_olsson_model( VectorType & solution_out );
    
    void 
    solve_olsson_model_matrixfree( VectorType & solution_out );

    void
    initialize_time_iterator(std::shared_ptr<TimeIterator> t);

    const MPI_Comm & mpi_commun;

    ReinitializationData                     reinit_data;
    bool                                     compute_normal_vector;

    SmartPointer<const DoFHandlerType>       dof_handler;
    SmartPointer<const ConstraintsType>      constraints;
    IndexSet                                 locally_owned_dofs;
    IndexSet                                 locally_relevant_dofs;

    SparseMatrixType                        system_matrix;
    VectorType                              system_rhs;
    ConditionalOStream                      pcout;
    NormalVector<dim>                       normal_vector_field;
    BlockVectorType                         solution_normal_vector;
  };
} // namespace LevelSetParallel
