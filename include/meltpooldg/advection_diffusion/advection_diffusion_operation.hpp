/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
// DoFTools
#include <deal.II/dofs/dof_tools.h>
// MeltPoolDG
#include <meltpooldg/advection_diffusion/advection_diffusion_operation_base.hpp>
#include <meltpooldg/advection_diffusion/advection_diffusion_operator.hpp>
#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/utilities/linearsolve.hpp>
#include <meltpooldg/utilities/utilityfunctions.hpp>

namespace MeltPoolDG
{
  namespace AdvectionDiffusion
  {
    using namespace dealii;

    template <int dim>
    class AdvectionDiffusionOperation : public AdvectionDiffusionOperationBase<dim>
    {
    private:
      using VectorType       = LinearAlgebra::distributed::Vector<double>;
      using BlockVectorType  = LinearAlgebra::distributed::BlockVector<double>;
      using SparseMatrixType = TrilinosWrappers::SparseMatrix;

    public:
      /*
       *  All the necessary parameters are stored in this struct.
       */
      AdvectionDiffusionData<double> advec_diff_data;

      AdvectionDiffusionOperation() = default;

      void
      initialize(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
                 const VectorType &                             solution_advected_field_in,
                 const Parameters<double> &                     data_in,
                 const unsigned int                             dof_idx_in,
                 const unsigned int                             dof_no_bc_idx_in,
                 const unsigned int                             quad_idx_in,
                 const unsigned int                             velocity_dof_idx_in) override
      {
        scratch_data     = scratch_data_in;
        dof_idx          = dof_idx_in;
        quad_idx         = quad_idx_in;
        dof_no_bc_idx    = dof_no_bc_idx_in;
        velocity_dof_idx = velocity_dof_idx_in;
        /*
         *  set the advection diffusion data
         */
        advec_diff_data = data_in.advec_diff;
        /*
         *  set the initial solution of the advected field
         */
        scratch_data->initialize_dof_vector(solution_advected_field, dof_idx);
        solution_advected_field.copy_locally_owned_data_from(solution_advected_field_in);
        solution_advected_field.update_ghost_values();
        /*
         *  set the parameters for the advection_diffusion problem
         */
        set_advection_diffusion_parameters(data_in);
      }


      void
      solve(const double dt, const BlockVectorType &advection_velocity) override
      {
        advection_velocity.update_ghost_values();

        create_operator(advection_velocity);

        VectorType src, rhs;

        scratch_data->initialize_dof_vector(src, dof_idx);
        scratch_data->initialize_dof_vector(rhs, dof_idx);

        advec_diff_operator->set_time_increment(dt);

        int iter = 0;

        if (advec_diff_data.do_matrix_free)
          {
            /*
             * apply dirichlet boundary values
             */
            advec_diff_operator->create_rhs_and_apply_dirichlet_mf(
              rhs, solution_advected_field, *scratch_data, dof_idx, dof_no_bc_idx);

            iter = LinearSolve<VectorType, SolverGMRES<VectorType>, OperatorBase<double>>::solve(
              *advec_diff_operator, src, rhs);
          }
        else
          {
            //@todo: which preconditioner?
            // TrilinosWrappers::PreconditionAMG preconditioner;
            // TrilinosWrappers::PreconditionAMG::AdditionalData data;

            // preconditioner.initialize(system_matrix, data);
            advec_diff_operator->assemble_matrixbased(solution_advected_field,
                                                      advec_diff_operator->system_matrix,
                                                      rhs);
            iter = LinearSolve<VectorType, SolverGMRES<VectorType>, SparseMatrixType>::solve(
              advec_diff_operator->system_matrix, src, rhs);
          }

        scratch_data->get_constraint(dof_idx).distribute(src);

        solution_advected_field = src;
        solution_advected_field.update_ghost_values();

        if (advec_diff_data.do_print_l2norm)
          {
            const ConditionalOStream &pcout = scratch_data->get_pcout();
            pcout << "| GMRES: i=" << std::setw(5) << std::left << iter;
            pcout << "\t |ϕ|2 = " << std::setw(15) << std::left << std::setprecision(10)
                  << solution_advected_field.l2_norm() << std::endl;
          }

        advection_velocity.zero_out_ghosts();
      }

      const LinearAlgebra::distributed::Vector<double> &
      get_advected_field() const override
      {
        return solution_advected_field;
      }

      LinearAlgebra::distributed::Vector<double> &
      get_advected_field() override
      {
        return solution_advected_field;
      }


      const LinearAlgebra::distributed::Vector<double> &
      get_advected_field_old() const override
      {
        return solution_advected_field;
      }

      LinearAlgebra::distributed::Vector<double> &
      get_advected_field_old() override
      {
        return solution_advected_field;
      }

      void
      attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) override
      {
        vectors.push_back(&solution_advected_field);
      }

    private:
      void
      set_advection_diffusion_parameters(const Parameters<double> &data_in)
      {
        advec_diff_data = data_in.advec_diff;
      }

      void
      create_operator(const BlockVectorType &advection_velocity)
      {
        advec_diff_operator = std::make_unique<AdvectionDiffusionOperator<dim, double>>(
          *scratch_data, advection_velocity, advec_diff_data, dof_idx, quad_idx, velocity_dof_idx);
        /*
         *  In case of a matrix-based simulation, setup the distributed sparsity pattern and
         *  apply it to the system matrix. This functionality is part of the OperatorBase class.
         */
        if (!advec_diff_data.do_matrix_free)
          advec_diff_operator->initialize_matrix_based<dim>(*scratch_data);
      }

    private:
      std::shared_ptr<const ScratchData<dim>> scratch_data;
      /*
       *  This pointer will point to your user-defined advection_diffusion operator.
       */
      std::unique_ptr<OperatorBase<double>> advec_diff_operator;
      /*
       *  Based on the following indices the correct DoFHandler or quadrature rule from
       *  ScratchData<dim> object is selected. This is important when ScratchData<dim> holds
       *  multiple DoFHandlers, quadrature rules, etc.
       */
      unsigned int dof_idx;
      unsigned int quad_idx;
      unsigned int dof_no_bc_idx;
      unsigned int velocity_dof_idx;
      /*
       *    This is the primary solution variable of this module, which will be also publically
       *    accessible for output_results.
       */
      VectorType solution_advected_field;
    };
  } // namespace AdvectionDiffusion
} // namespace MeltPoolDG
