#pragma once
// for distributed vectors/matrices
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
// constraints
#include <deal.II/lac/affine_constraints.h>
// solvers
#include <deal.II/lac/solver_cg.h> // only for symmetric matrices
#include <deal.II/lac/solver_gmres.h>

// preconditioner
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>

using namespace dealii;

namespace LevelSetParallel
{

template<typename VectorType, 
         typename SolverType         = SolverGMRES<VectorType>, 
         typename OperatorType       = TrilinosWrappers::SparseMatrix, 
         typename PreconditionerType = PreconditionIdentity>
class LinearSolve
{
  private:
    typedef AffineConstraints<double>         ConstraintsType;

  public:
    static int solve( const OperatorType&       system_matrix,
                       VectorType&               solution,
                       const VectorType&         rhs, 
                       const PreconditionerType& preconditioner    = PreconditionIdentity(),
                       const unsigned int        max_iterations    = 10000,
                       const double              rel_tolerance_rhs = 1e-8
                    )
    {
      SolverControl   solver_control( max_iterations, rel_tolerance_rhs * rhs.l2_norm() );
      SolverType      solver(         solver_control );

      solver.solve( system_matrix, 
                    solution, 
                    rhs, 
                    preconditioner);
      
      solution.update_ghost_values();
      return solver_control.last_step();
    }
    
};

} // LevelSetParallel
