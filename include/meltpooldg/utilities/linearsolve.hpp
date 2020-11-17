#pragma once
// for distributed vectors/matrices
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
// solvers
#include <deal.II/lac/solver_cg.h> // only for symmetric matrices
#include <deal.II/lac/solver_gmres.h>

// preconditioner
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>


using namespace dealii;

namespace MeltPoolDG
{
  template <typename VectorType,
            typename SolverType         = SolverGMRES<VectorType>,
            typename OperatorType       = TrilinosWrappers::SparseMatrix,
            typename PreconditionerType = PreconditionIdentity>
  class LinearSolve
  {
  public:
    static int
    solve(const OperatorType &      system_matrix,
          VectorType &              solution,
          const VectorType &        rhs,
          const PreconditionerType &preconditioner    = PreconditionIdentity(),
          const unsigned int        max_iterations    = 10000,
          const double              rel_tolerance_rhs = 1e-8)
    {
      SolverControl solver_control(max_iterations, rel_tolerance_rhs * rhs.l2_norm());
      SolverType    solver(solver_control);

      solver.solve(system_matrix, solution, rhs, preconditioner);

      solution.update_ghost_values();
      return solver_control.last_step();
    }

    static std::shared_ptr<TrilinosWrappers::PreconditionBase>
    setup_preconditioner(const TrilinosWrappers::SparseMatrix& matrix, std::string preconditioner_type="Identity")
    {
      if (preconditioner_type=="Identity")
      {
        return std::make_shared<TrilinosWrappers::PreconditionIdentity>();
      }
      else if (preconditioner_type=="AMG")
      {
        auto preconditioner = std::make_shared<TrilinosWrappers::PreconditionAMG>();
        TrilinosWrappers::PreconditionAMG::AdditionalData data;
        preconditioner->initialize(matrix,data);
        return preconditioner;
      }
      else if (preconditioner_type=="ILU")
      {
        auto preconditioner = std::make_shared<TrilinosWrappers::PreconditionILU>();
        TrilinosWrappers::PreconditionILU::AdditionalData data;
        preconditioner->initialize(matrix,data);
        return preconditioner;
      }
      else
        AssertThrow(false, ExcMessage("The requested preconditioner type is not implemented."))
    }
  };

} // namespace MeltPoolDG
