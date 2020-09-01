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
//#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
using namespace dealii;

namespace LevelSetParallel
{

template<typename VectorType, typename SolverType=SolverGMRES<VectorType>, typename OperatorType=TrilinosWrappers::SparseMatrix, typename PreconditionerType=PreconditionIdentity>
class LinearSolve
{
  private:
    typedef TrilinosWrappers::SparseMatrix    SparseMatrixType;
    typedef AffineConstraints<double>         ConstraintsType;

  public:
    static void solve( const OperatorType&       system_matrix,
                       const VectorType&         rhs, 
                       VectorType&               solution,
                       const ConstraintsType&    constraints,
                       const MPI_Comm&           mpi_communicator,
                       const IndexSet&           locally_owned_dofs,
                       const unsigned int        max_iterations    = 1000,
                       const double              rel_tolerance_rhs = 1e-8, 
                       const bool                print_iterations  = true,
                       const PreconditionerType& preconditioner    = PreconditionIdentity())
    {
      SolverControl           solver_control( max_iterations, rel_tolerance_rhs * rhs.l2_norm() );
      SolverType solver( solver_control );

      // @ --> solution directly without completely_distributed_solution??
      //VectorType    completely_distributed_solution( locally_owned_dofs,
                                                     //mpi_communicator);

      solver.solve( system_matrix, 
                    solution, 
                    rhs, 
                    preconditioner);
      //solution = completely_distributed_solution;
      solution.update_ghost_values();
      if (print_iterations && Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
          std::cout << " equation: " << solver_control.last_step() << "iterations." << std::endl;
    }
    
};

} // LevelSetParallel
