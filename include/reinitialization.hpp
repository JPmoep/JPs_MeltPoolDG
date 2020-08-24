/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, August 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

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
    typedef LA::MPI::Vector                           VectorType;
    typedef LA::MPI::BlockVector                      BlockVectorType;
    typedef LA::MPI::SparseMatrix                     SparseMatrixType;

    typedef DoFHandler<dim>                           DoFHandlerType;
    
    typedef DynamicSparsityPattern                    SparsityPatternType;
    
    typedef AffineConstraints<double>                 ConstraintsType;

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

    const MPI_Comm & mpi_commun;

    ReinitializationData                     reinit_data;
    bool                                     compute_normal_vector;

    SmartPointer<const DoFHandlerType>       dof_handler;
    SmartPointer<const ConstraintsType>      constraints;
    IndexSet                                 locally_owned_dofs;
    IndexSet                                 locally_relevant_dofs;

    SparseMatrixType      system_matrix;
    VectorType            system_rhs;
    ConditionalOStream    pcout;
    NormalVector<dim>     normal_vector_field;
    BlockVectorType       solution_normal_vector;
  };
} // namespace LevelSetParallel
