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
#include <deal.II/fe/mapping.h>
#include <deal.II/base/table_handler.h>
// MeltPoolDG
#include "utilityfunctions.hpp"
#include "normalvector.hpp"
#include "timeiterator.hpp"
#include "problembase.hpp"
#include "simulationbase.hpp"

namespace MeltPoolDG
{
  using namespace dealii; 

  /*
   *    Data for reinitialization of level set equation
   */
  
  enum class ReinitModelType { olsson2007=1, 
                               undefined=0  };
  
  struct ReinitializationData
  {
    ReinitializationData()
        : reinit_model(      ReinitModelType::undefined)
        , d_tau(             0.01)
        , constant_epsilon( -1.0)
        , max_reinit_steps(  5)
        , do_print_l2norm(   false)
        , do_matrix_free(    false)
        , verbosity_level(   TypeDefs::VerbosityType::silent)
    {
    }
    
    // enum which reinitialization model should be solved
    ReinitModelType reinit_model;
    
    // time step for reinitialization
    double d_tau;
    
    // choose a constant, not cell-size dependent smoothing parameter
    double constant_epsilon;

    // maximum number of reinitialization steps to be completed
    unsigned int max_reinit_steps;
    
    // this parameter controls whether the l2 norm is printed (mainly for testing purposes)
    bool do_print_l2norm;
    
    // this parameter activates the matrix free cell loop procedure
    bool do_matrix_free;
    
    // maximum number of reinitialization steps to be completed
    TypeDefs::VerbosityType verbosity_level;

    // @ add lambda function for calculating epsilon
  };
  
  /*
   *     Reinitialization model for reobtaining the signed-distance 
   *     property of the level set equation
   */
  
  template <int dim, int degree>
  class Reinitialization : public ProblemBase<dim,degree>
  {
  private:
      
    typedef LinearAlgebra::distributed::Vector<double>      VectorType;
    typedef LinearAlgebra::distributed::BlockVector<double> BlockVectorType;
    typedef TrilinosWrappers::SparseMatrix                  SparseMatrixType;
    typedef TrilinosWrappers::SparsityPattern               SparsityPatternType;

    typedef DoFHandler<dim>                                 DoFHandlerType;
    typedef AffineConstraints<double>                       ConstraintsType;

  public:

    /*
     *  Constructor as main module
     */
    Reinitialization( std::shared_ptr<SimulationBase<dim>> base );
    /*
     *  Usage as module: this function is the "global" run function to be called from the problem base class
     */
    void 
    run(); 
    /*
     *  Usage as submodule
     */
    Reinitialization(const MPI_Comm & mpi_communicator_in);
    
    /*
     *  Usage as submodule
     */
    void
    initialize_as_submodule( const ReinitializationData& data_in,
                             const SparsityPatternType&  dsp_in,
                             const DoFHandlerType&       dof_handler_in,
                             const ConstraintsType&      constraints_in,
                             const IndexSet&             locally_owned_dofs_in,
                             const IndexSet&             locally_relevant_dofs_in,
                             const double                min_cell_size_in);

    /*
     *  Usage as submodule: This function reinitializes the solution of the level set equation for a given solution
     */
    void 
    run_as_submodule( VectorType & solution_out );

    void 
    print_me(); 
    
    // @ does this really need to be global??
    void 
    initialize_data_from_global_parameters(const Parameters& data_in); 
    
    /*
     *  this function returns the last calculated normal vector
     */
    BlockVectorType
    get_normal_vector_field() const; 
    
    std::string get_name() final { return "reinitialization"; };

  private:
    /*
     *  Usage as module: this function initials the relevant member data
     *  for the computation of the module
     */
    void 
    initialize_module();
    /*
     *  Usage as module: this function is for creating paraview output
     */
    void 
    output_results(const VectorType& solution, const double time=0.0); 
    /* 
     * This function solves the Olsson, Kreiss, Zahedi (2007) model for reinitialization 
     * of the level set equation.
     */
    void 
    solve_olsson_model( VectorType & solution_out );
    /* 
     * This function is a reimplementation of solve_olsson_model using matrixfree operators.
     * An input parameter is 
     * Olsson, Kreiss, Zahedi (2007) model 
     * for reinitialization of the level set equation 
     */
    void 
    solve_olsson_model_matrixfree( VectorType & solution_out );

    void
    initialize_time_iterator(std::shared_ptr<TimeIterator> t);

  
    // for submodule this is actually needed as reference
    const MPI_Comm                             mpi_communicator;
    ConditionalOStream                         pcout;
    DoFHandlerType                             module_dof_handler;
    std::shared_ptr<FieldConditions<dim>>      field_conditions;
    NormalVector<dim,degree>                   normal_vector_field;
    double                                     min_cell_size;     // @todo: check CFL condition
    
    ConstraintsType                            module_constraints;
    Parameters                                 parameters;
    ReinitializationData                       reinit_data;
    /* 
    * at the moment the implementation considers natural boundary conditions
     */
    //std::shared_ptr<BoundaryConditions<dim>>   boundary_conditions;
    
    /*
    * the following two could be larger objects, thus we do not want
    * to copy them in the case of usage as a submodule
    */
    SmartPointer<const DoFHandlerType>      dof_handler;
    SmartPointer<const ConstraintsType>     constraints;
    IndexSet                                locally_owned_dofs;
    IndexSet                                locally_relevant_dofs;
  
    SparsityPatternType                     dsp;
    SparseMatrixType                        system_matrix;     // @todo: might not be a member variable
    VectorType                              system_rhs;        // @todo: might not be member variables
    BlockVectorType                         solution_normal_vector;
    TableHandler                            table;
  };
} // namespace MeltPoolDG
