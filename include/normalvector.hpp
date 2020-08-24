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

#include <utilityFunctions.hpp>

namespace LevelSetParallel
{
  using namespace dealii; 

  /*
   *    Data for computing the normal vector of a given scalar field 
   *    considering diffusive damping;
   *
   *    !!!! 
   *          the normal vector field is not normalized to length one, 
   *          it actually represents the gradient of the level set 
   *          function 
   *    !!!! 
   */
  
  struct NormalVectorData 
  {
    NormalVectorData()
        : damping_parameter(0.0)
        , degree(1)
        , verbosity_level(utilityFunctions::VerbosityType::silent)
    {
    }
    
    // parameter for diffusive term in computation of normals
    double damping_parameter;
    
    // interpolation degree of normal vector interpolation
    unsigned int degree;

    // current verbosity level --> see possible options in utilityFunctions
    utilityFunctions::VerbosityType verbosity_level;
  };
  
  /*
   *     Model for computing the normal vector to a scalar function as a smooth function
   *     @ todo: add equation 
   */
  
  template <int dim>
  class NormalVector
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
    NormalVector(const MPI_Comm & mpi_commun_in);

    void
    initialize( const NormalVectorData&          data_in,
                const SparsityPatternType&       dsp_in,
                DoFHandler<dim> const &          dof_handler_in,
                const ConstraintsType&           constraints_in,
                const IndexSet&                  locally_owned_dofs_in);

    /*
     *  This function computes the (damped) normal vector field for a given solution of a scalar function
     */
    void 
    compute_normal_vector_field( const VectorType & level_set_solution_in,
                                 BlockVectorType & normal_vector_out);

    void 
    print_me(); 
  
  private:
    void 
    solve_cg( VectorType & solution, const VectorType & rhs );

    const MPI_Comm & mpi_commun;

    NormalVectorData                        normal_vector_data;
    SmartPointer<const DoFHandlerType>      dof_handler;
    SmartPointer<const ConstraintsType>     constraints;
    IndexSet                                locally_owned_dofs;

    SparseMatrixType                        system_matrix;
    BlockVectorType                         system_rhs;
    //BlockVectorType                         normal_vector_field;
    ConditionalOStream                      pcout;
  };
} // namespace LevelSetParallel
