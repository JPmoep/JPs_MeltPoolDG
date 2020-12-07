/* ---------------------------------------------------------------------
 *
 * Author: Peter Münch, Magdalena Schreter, TUM, December 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

#ifdef MELT_POOL_DG_WITH_ADAFLO

#  include <deal.II/lac/generic_linear_algebra.h>

#  include <meltpooldg/flow/adaflo_wrapper_parameters.hpp>
#  include <meltpooldg/flow/flow_base.hpp>
#  include <meltpooldg/interface/scratch_data.hpp>
#  include <meltpooldg/utilities/vector_tools.hpp>

#  include <adaflo/diagonal_preconditioner.h>
#  include <adaflo/level_set_okz_advance_concentration.h>

namespace MeltPoolDG
{
  namespace AdvectionDiffusionAdaflo
  {
    template <int dim>
    class AdafloWrapper
    {
    private:
      using VectorType      = LinearAlgebra::distributed::Vector<double>;
      using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    public:
      /**
       * Constructor.
       */
      AdafloWrapper(ScratchData<dim> &                   scratch_data,
                    const int                            advec_diff_dof_idx,
                    const int                            advec_diff_quad_idx,
                    const VectorType &                   solution_advected_field,
                    BlockVectorType &                    velocity_vec_in,
                    std::shared_ptr<SimulationBase<dim>> base_in)
        : scratch_data(scratch_data)
        , advec_diff_dof_idx(advec_diff_dof_idx)
        , advec_diff_quad_idx(advec_diff_quad_idx)
        , advected_field(solution_advected_field)
        , velocity_vec(velocity_vec_in)
      {
        scratch_data.initialize_dof_vector(rhs);
        scratch_data.initialize_dof_vector(increment);

        compute_velocity_quadrature();
        initialize_preconditioner();

        advec_diff_operation = std::make_shared<LevelSetOKZSolverAdvanceConcentration<dim>>(
          advected_field,
          advected_field_old,
          advected_field_old_old,
          increment,
          rhs,
          velocity_vec_temp, //@todo: convert block vector to fe_system vector
          velocity_vec_temp, //@todo: convert block vector to fe_system vector
          velocity_vec_temp, //@todo: convert block vector to fe_system vector
          scratch_data.get_diameter(advec_diff_dof_idx), /*global_omega_diameter*/
          scratch_data.get_cell_diameters(advec_diff_dof_idx,
                                          advec_diff_quad_idx,
                                          true), /*global_omega_diameter*/
          scratch_data.get_constraint(advec_diff_dof_idx),
          scratch_data.get_pcout(advec_diff_dof_idx),
          bcs,
          scratch_data.get_matrix_free(),
          adaflo_params,
          global_max_velocity,
          preconditioner,
          velocities);

        /*
         * Boundary conditions for the advected field
         */
        //@todo
        /*
         * Initial conditions for the advected field
         */
        //@todo
      }

      /**
       * Solver time step
       */
      void
      solve(const double dt)
      {
        advected_field_old     = advected_field;
        advected_field_old_old = advected_field_old;

        compute_velocity_quadrature();
        advec_diff_operation->advance_concentration(dt);
      }

      const LinearAlgebra::distributed::Vector<double> &
      get_advected_field() const
      {
        return advected_field;
      }

    private:
      void
      compute_velocity_quadrature()
      {
        FECellIntegrator<dim, 1, double> fe_eval(scratch_data.get_matrix_free(),
                                                 advec_diff_dof_idx,
                                                 advec_diff_quad_idx);

        for (unsigned int cell = 0; cell < scratch_data.get_matrix_free().n_cell_batches(); ++cell)
          {
            fe_eval.reinit(cell);
            fe_eval.read_dof_values_plain(velocity_vec);

            for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
              velocities[cell * fe_eval.n_q_points + q] = fe_eval.get_value(q);
          }
      }

      void
      initialize_preconditioner()
      {
        // create diagonal preconditioner vector by assembly of mass matrix diagonal
        LinearAlgebra::distributed::Vector<double> diagonal(advected_field);
        diagonal = 1.0;
        preconditioner.reinit(diagonal);
      }

      /**
       * Boundary conditions for the advection diffusion operation
       * @todo
       */
      LevelSetOKZSolverAdvanceConcentrationBoundaryDescriptor<dim> bcs;
      LevelSetOKZSolverAdvanceConcentrationParameter               adaflo_params;

      /**
       * Reference to the actual Navier-Stokes solver from adaflo
       */
      std::shared_ptr<LevelSetOKZSolverAdvanceConcentration<dim>> advec_diff_operation;

      /**
       * artificial_viscosities ->set b< adaflo
       */
      AlignedVector<VectorizedArray<double>> artificial_viscosities;
      /**
       *  maximum velocity --> set by adaflo
       */
      double global_max_velocity;
      /**
       *  velocity values at gauss points
       */
      AlignedVector<Tensor<1, dim, VectorizedArray<double>>> velocities;
      /**
       *  Diagonal preconditioner
       */
      DiagonalPreconditioner<double> preconditioner;
      BlockVectorType &              velocity_vec;
      VectorType                     velocity_vec_temp;
      /**
       *  advected field
       */
      VectorType &advected_field;
      VectorType  advected_field_old;
      VectorType  advected_field_old_old;
      VectorType  increment;
      VectorType  rhs;

      const int advec_diff_dof_idx;
      const int advec_diff_quad_idx;

      const ScratchData<dim> &scratch_data;
    };

    /**
     * Dummy specialization for 1. Needed to be able to compile
     * since the adaflo Navier-Stokes solver is not compiled for
     * 1D - due to the dependcy to parallel::distributed::Triangulation
     * and p4est.
     */
    // @todo
  } // namespace AdvectionDiffusionAdaflo
} // namespace MeltPoolDG

#endif
