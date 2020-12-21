/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, August 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
// for using smart pointers
#include <deal.II/base/smartpointer.h>

// MeltPoolDG
#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/normal_vector/normal_vector_operation.hpp>
#include <meltpooldg/normal_vector/normal_vector_operation_adaflo_wrapper.hpp>
#include <meltpooldg/reinitialization/olsson_operator.hpp>
#include <meltpooldg/reinitialization/reinitialization_operation_base.hpp>
#include <meltpooldg/utilities/linearsolve.hpp>
#include <meltpooldg/utilities/utilityfunctions.hpp>

namespace MeltPoolDG
{
  namespace Reinitialization
  {
    using namespace dealii;

    /*
     *     Reinitialization model for reobtaining the signed-distance
     *     property of the level set equation
     */

    template <int dim>
    class ReinitializationOperation : public ReinitializationOperationBase<dim>
    {
    private:
      using VectorType       = LinearAlgebra::distributed::Vector<double>;
      using BlockVectorType  = LinearAlgebra::distributed::BlockVector<double>;
      using SparseMatrixType = TrilinosWrappers::SparseMatrix;

    public:
      ReinitializationData<double> reinit_data;
      /*
       *    This is the primary solution variable of this module, which will be also publically
       *    accessible for output_results.
       */
      VectorType solution_level_set;

      ReinitializationOperation() = default;

      void
      initialize(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
                 const VectorType &                             solution_level_set_in,
                 const Parameters<double> &                     data_in,
                 const unsigned int                             reinit_dof_idx_in,
                 const unsigned int                             reinit_quad_idx_in,
                 const unsigned int                             normal_dof_idx_in) override
      {
        scratch_data    = scratch_data_in;
        reinit_dof_idx  = reinit_dof_idx_in;
        reinit_quad_idx = reinit_quad_idx_in;
        normal_dof_idx  = normal_dof_idx_in;
        scratch_data->initialize_dof_vector(solution_level_set, reinit_dof_idx);
        /*
         *    initialize the (local) parameters of the reinitialization
         *    from the global user-defined parameters
         */
        set_reinitialization_parameters(data_in);
        /*
         *    initialize normal_vector_field
         */
        AssertThrow(data_in.normal_vec.do_matrix_free == data_in.reinit.solver.do_matrix_free,
                    ExcMessage("For the reinitialization problem both the "
                               " normal vector and the reinitialization operation have to be "
                               " computed either matrix-based or matrix-free."));

        if (data_in.normal_vec.implementation == "meltpooldg")
          {
            normal_vector_operation = std::make_shared<NormalVector::NormalVectorOperation<dim>>();

            normal_vector_operation->initialize(
              scratch_data_in, data_in, normal_dof_idx, reinit_quad_idx, reinit_dof_idx);
          }
#ifdef MELT_POOL_DG_WITH_ADAFLO
        else if (data_in.normal_vec.implementation == "adaflo")
          {
            AssertThrow(data_in.normal_vec.do_matrix_free, ExcNotImplemented());

            normal_vector_operation =
              std::make_shared<NormalVector::NormalVectorOperationAdaflo<dim>>(
                *scratch_data_in,
                reinit_dof_idx, //@todo -- this is actually ls dof idx; must this be added??
                normal_dof_idx,
                reinit_quad_idx,
                solution_level_set,
                data_in);
          }
#endif
        else
          AssertThrow(false, ExcNotImplemented());
        /*
         *    compute the normal vector field and update the initial solution
         */
        update_initial_solution(solution_level_set_in);
        /*
         *   create reinitialization operator. This class supports matrix-based
         *   and matrix-free computation.
         */
        create_operator();
      }

      void
      reinit() override
      {
        scratch_data->initialize_dof_vector(solution_level_set, reinit_dof_idx);
      }

      /*
       *  By calling the reinitialize function, (1) the solution_level_set field
       *  and (2) the normal vector field corresponding to the given solution_level_set_field
       *  is updated. This is commonly the first stage before performing the pseudo-time-dependent
       *  solution procedure.
       */
      void
      update_initial_solution(const VectorType &solution_level_set_in) override
      {
        /*
         *    copy the given solution into the member variable
         */
        scratch_data->initialize_dof_vector(solution_level_set, reinit_dof_idx);
        solution_level_set.copy_locally_owned_data_from(solution_level_set_in);
        solution_level_set.update_ghost_values();
        /*
         *    update the normal vector field corresponding to the given solution of the
         *    level set; the normal vector field is called by reference within the
         *    operator class
         */
        normal_vector_operation->reinit();
        normal_vector_operation->solve(solution_level_set);
      }

      void
      solve(const double d_tau) override
      {
        /**
         * update the distributed sparsity pattern for matrix-based amr
         */
        update_operator();
        VectorType src, rhs;

        scratch_data->initialize_dof_vector(src, reinit_dof_idx);
        scratch_data->initialize_dof_vector(rhs, reinit_dof_idx);

        reinit_operator->set_time_increment(d_tau);

        int iter = 0;

        if (reinit_data.solver.do_matrix_free)
          {
            VectorType src_rhs;
            scratch_data->initialize_dof_vector(src_rhs, reinit_dof_idx);
            src_rhs.copy_locally_owned_data_from(solution_level_set);
            src_rhs.update_ghost_values();
            reinit_operator->create_rhs(rhs, src_rhs);
            iter = LinearSolve<VectorType, SolverCG<VectorType>, OperatorBase<double>>::solve(
              *reinit_operator, src, rhs);
          }
        else
          {
            reinit_operator->system_matrix.reinit(reinit_operator->dsp);
            reinit_operator->assemble_matrixbased(solution_level_set,
                                                  reinit_operator->system_matrix,
                                                  rhs);

            if (reinit_data.solver.solver_type == "CG")
              {
                auto preconditioner = LinearSolve<VectorType,
                                                  SolverCG<VectorType>,
                                                  SparseMatrixType,
                                                  TrilinosWrappers::PreconditionBase>::
                  setup_preconditioner(reinit_operator->system_matrix,
                                       reinit_data.solver.preconditioner_type);
                iter = LinearSolve<
                  VectorType,
                  SolverCG<VectorType>,
                  SparseMatrixType,
                  TrilinosWrappers::PreconditionBase>::solve(reinit_operator->system_matrix,
                                                             src,
                                                             rhs,
                                                             *preconditioner,
                                                             reinit_data.solver.max_iterations,
                                                             reinit_data.solver.rel_tolerance_rhs);
              }
            else if (reinit_data.solver.solver_type == "GMRES")
              {
                auto preconditioner = LinearSolve<VectorType,
                                                  SolverGMRES<VectorType>,
                                                  SparseMatrixType,
                                                  TrilinosWrappers::PreconditionBase>::
                  setup_preconditioner(reinit_operator->system_matrix,
                                       reinit_data.solver.preconditioner_type);
                iter = LinearSolve<
                  VectorType,
                  SolverGMRES<VectorType>,
                  SparseMatrixType,
                  TrilinosWrappers::PreconditionBase>::solve(reinit_operator->system_matrix,
                                                             src,
                                                             rhs,
                                                             *preconditioner,
                                                             reinit_data.solver.max_iterations,
                                                             reinit_data.solver.rel_tolerance_rhs);
              }
          }
        scratch_data->get_constraint(reinit_dof_idx).distribute(src);

        solution_level_set += src;

        solution_level_set.update_ghost_values();

        if (reinit_data.do_print_l2norm)
          {
            const ConditionalOStream &pcout = scratch_data->get_pcout(reinit_dof_idx);
            pcout << "| CG: i=" << std::setw(5) << std::left << iter;
            pcout << "\t |ΔΨ|∞ = " << std::setw(15) << std::left << std::setprecision(10)
                  << src.linfty_norm();
            pcout << " |ΔΨ|²/dT = " << std::setw(15) << std::left << std::setprecision(10)
                  << src.l2_norm() / d_tau << "|" << std::endl;
          }
      }

      const BlockVectorType &
      get_normal_vector() const override
      {
        return normal_vector_operation->get_solution_normal_vector();
      }

      const VectorType &
      get_level_set() const override
      {
        return solution_level_set;
      }

      VectorType &
      get_level_set() override
      {
        return solution_level_set;
      }

      void
      attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
      {
        vectors.push_back(&solution_level_set);
      }


    private:
      void
      set_reinitialization_parameters(const Parameters<double> &data_in)
      {
        reinit_data = data_in.reinit;
      }

      void
      create_operator()
      {
        if (reinit_data.modeltype == "olsson2007")
          {
            reinit_operator = std::make_unique<OlssonOperator<dim, double>>(
              *scratch_data,
              normal_vector_operation->get_solution_normal_vector(),
              reinit_data.constant_epsilon,
              reinit_data.scale_factor_epsilon,
              reinit_dof_idx,
              reinit_quad_idx);
          }
        /*
         * add your desired operators here
         *
         * else if (reinit_data.reinitmodel == "my_model")
         *    ....
         */
        else
          AssertThrow(false, ExcMessage("Requested reinitialization model not implemented."))
            /*
             *  In case of a matrix-based simulation, setup the distributed sparsity pattern and
             *  apply it to the system matrix. This functionality is part of the OperatorBase class.
             */

            if (!reinit_data.solver.do_matrix_free)
              reinit_operator->initialize_matrix_based<dim>(*scratch_data);
      }
      void
      update_operator()
      {
        if (!reinit_data.solver.do_matrix_free)
          reinit_operator->initialize_matrix_based<dim>(*scratch_data);
      }



    private:
      std::shared_ptr<const ScratchData<dim>> scratch_data;
      /*
       *  This shared pointer will point to your user-defined reinitialization operator.
       */
      std::unique_ptr<OperatorBase<double>> reinit_operator;
      /*
       *   Computation of the normal vectors
       */
      std::shared_ptr<NormalVector::NormalVectorOperationBase<dim>> normal_vector_operation;
      // NormalVector::NormalVectorOperation<dim> normal_vector_operation;
      /*
       *  Based on the following indices the correct DoFHandler or quadrature rule from
       *  ScratchData<dim> object is selected. This is important when ScratchData<dim> holds
       *  multiple DoFHandlers, quadrature rules, etc.
       */
      unsigned int reinit_dof_idx;
      unsigned int reinit_quad_idx;
      unsigned int normal_dof_idx;
    };
  } // namespace Reinitialization
} // namespace MeltPoolDG
