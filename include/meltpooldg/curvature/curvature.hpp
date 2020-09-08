/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, August 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
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
// for dof_handler type
#include <deal.II/dofs/dof_handler.h>
// for FE_Q<dim> type
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
// for FE_Q<dim> type
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

// from MeltPoolDG
#include "utilityfunctions.hpp"
#include "normalvector.hpp"

namespace MeltPoolDG
{
  using namespace dealii; 

  /*
   *    Data for computing the curvature of a given level set function
   */
  
  struct CurvatureData
  {
    CurvatureData()
        : damping_parameter(0.0)
        , min_cell_size(0.0)
        , verbosity_level(TypeDefs::VerbosityType::silent)
    {
    }
    
    // parameter for diffusive term in computation of normals
    double damping_parameter;
    
    // minimum size of cells --> to evaluate damping parameter @todo: should this parameter be made cell-size-dependent?
    double min_cell_size;

    // current verbosity level --> see possible options in utilityFunctions
    TypeDefs::VerbosityType verbosity_level;
  };
  
  /*
   *     Curvature model
   */
  
  template <int dim, int degree>
  class Curvature
  {
  private:
    using VectorType          = LinearAlgebra::distributed::Vector<double>;         
    using BlockVectorType     = LinearAlgebra::distributed::BlockVector<double>;    
    using SparseMatrixType    = TrilinosWrappers::SparseMatrix;                     
    using DoFHandlerType      = DoFHandler<dim>;                                    
    using SparsityPatternType = TrilinosWrappers::SparsityPattern;
    using ConstraintsType     = AffineConstraints<double>;                          

  public:

    Curvature(const MPI_Comm & mpi_commun_in);

    void
    initialize( const CurvatureData &       data_in,
                const SparsityPatternType&  dsp_in,
                DoFHandler<dim> const &     dof_handler_in,
                const ConstraintsType&      constraints_in,
                const IndexSet&             locally_owned_dofs_in,
                const IndexSet&             locally_relevant_dofs_in);

    /*
     *  This function calculates the curvature of the current level set function being
     *  the solution of an intermediate projection step 
     *   
     *              (w, κ)   +   η_κ (∇w, ∇κ)  = (w,∇·n_ϕ)
     *                    Ω                  Ω            Ω            
     *  
     *  with test function w, curvature κ, damping parameter η_κ and the normal to the
     *  level set function n_ϕ.
     *
     */
    void 
    solve( const VectorType & levelset_solution_in,
                 VectorType & curvature_solution_out );
    
    void 
    solve( const VectorType & levelset_solution_in );
    
    VectorType 
    get_curvature_values() const; 

    void 
    print_me(); 
  
  private:
    void 
    solve_cg( VectorType & solution, const VectorType & rhs );
    
    const MPI_Comm & mpi_commun;

    CurvatureData                            curvature_data;

    SmartPointer<const DoFHandlerType>       dof_handler;
    SmartPointer<const ConstraintsType>      constraints;
    IndexSet                                 locally_owned_dofs;
    IndexSet                                 locally_relevant_dofs;

    SparseMatrixType                         system_matrix;
    VectorType                               system_rhs;
    ConditionalOStream                       pcout;
    VectorType                               curvature_field;
    NormalVector<dim,degree>                 normal_vector_field;
  };
} // namespace MeltPoolDG
