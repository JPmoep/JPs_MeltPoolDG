/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, August 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
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

// MeltPoolDG
#include "utilityfunctions.hpp"
#include "parameters.hpp"

namespace MeltPoolDG
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
   *
   *    This module is a sub-module and cannot be instantiated as a 
   *    module. 
   */
  
  struct NormalVectorData 
  {
    NormalVectorData()
        : damping_parameter(1e-6)
        , do_print_l2norm(false)
        , verbosity_level(TypeDefs::VerbosityType::silent)
    {
    }

    // parameter for diffusive term in computation of normals
    double damping_parameter;
    
    // this parameter controls whether the l2 norm is printed (mainly for testing purposes)
    bool do_print_l2norm;
    
    // current verbosity level --> see possible options in UtilityFunctions
    TypeDefs::VerbosityType verbosity_level;

  };
  
  /*
   *     Model for computing the normal vector to a scalar function as a smooth function
   *     @ todo: add equation 
   */
  
  template <int dim, int degree>
  class NormalVector
  {
  private:

    typedef LinearAlgebra::distributed::Vector<double>      VectorType;
    typedef LinearAlgebra::distributed::BlockVector<double> BlockVectorType;
    typedef TrilinosWrappers::SparseMatrix                  SparseMatrixType;

    typedef DoFHandler<dim>                                 DoFHandlerType;
    
    typedef TrilinosWrappers::SparsityPattern               SparsityPatternType;
    
    typedef AffineConstraints<double>                       ConstraintsType;

  public:

    /*
     *  Constructor
     */
    NormalVector(const MPI_Comm & mpi_commun_in);
  

    void
    initialize( const NormalVectorData&     data_in,
                const SparsityPatternType&  dsp_in,
                const DoFHandlerType&       dof_handler_in,
                const ConstraintsType&      constraints_in,
                const IndexSet&             locally_owned_dofs_in,
                const IndexSet&             locally_relevant_dofs_in,
                const double                min_cell_size_in);
    
    /*
     *  This function initializes the values of NormalVectorData considering the input parameters of the 
     *  SimulationBase object. 
     */ 
    void 
    extract_local_parameters_from_global_parameters( const Parameters& level_set_solution_in);
    
    
    void 
    compute_normal_vector_field_matrixfree( const VectorType & levelset_in,
                                            BlockVectorType & normal_vector_out); 
    /*
     *  This function overloads the previous one, where the normal vectors are stored as a member variable.
     */ 
    void 
    compute_normal_vector_field_matrixfree( const VectorType & level_set_solution_in);

    /*
     *  This function computes the (damped) normal vector field for a given solution of a scalar function
     */
    void 
    compute_normal_vector_field( const VectorType & level_set_solution_in,
                                 BlockVectorType & normal_vector_out);
    
    /*
     *  This function overloads the previous one, where the normal vectors are stored as a member variable.
     */ 
    void 
    compute_normal_vector_field( const VectorType & level_set_solution_in);

    void 
    print_me(); 
    
    /*
     *  For a given (non-normalized) vector field calculate the (normalized) vector field at the quadrature
     *  points.
     */

    void
    get_unit_normals_at_quadrature( const FEValues<dim>& fe_values,
                                    const BlockVectorType& normal_vector_in, 
                                    std::vector<Tensor<1,dim>>& normal_at_gauss ) const;
    /*
     * This function overloads the previous function by using the stored vector field as a member variable
     */
    void
    get_unit_normals_at_quadrature( const FEValues<dim>& fe_values,
                                    std::vector<Tensor<1,dim>>& normal_at_gauss ) const;
  
  private:
    const MPI_Comm & mpi_commun;

    NormalVectorData                        normal_vector_data;
    SmartPointer<const DoFHandlerType>      dof_handler;
    SmartPointer<const ConstraintsType>     constraints;
    IndexSet                                locally_owned_dofs;
    IndexSet                                locally_relevant_dofs;
    double                                  min_cell_size;

    SparseMatrixType                        system_matrix;
    BlockVectorType                         system_rhs;
    BlockVectorType                         normal_vector_field;
    ConditionalOStream                      pcout;
  };
} // namespace MeltPoolDG
