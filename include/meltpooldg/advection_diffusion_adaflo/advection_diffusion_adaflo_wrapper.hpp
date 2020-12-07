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
                    const int                            velocity_dof_idx,
                    VectorType &                         solution_advected_field,
                    BlockVectorType &                    velocity_vec_in, // @todo: make const ref
                    std::shared_ptr<SimulationBase<dim>> base_in)
        : scratch_data(scratch_data)
        , advected_field(solution_advected_field)
        , velocity_vec(velocity_vec_in)
      {
        scratch_data.initialize_dof_vector(rhs);
        scratch_data.initialize_dof_vector(increment);
        scratch_data.initialize_dof_vector(advected_field_old);
        scratch_data.initialize_dof_vector(advected_field_old_old);
        scratch_data.initialize_dof_vector(velocity_vec_temp, velocity_dof_idx);

        set_velocity(velocity_vec);

        initialize_preconditioner();

        /*
         * set parameters of adaflo
         * @todo
         */
        adaflo_params.dof_index_ls  = advec_diff_dof_idx;
        adaflo_params.dof_index_vel = velocity_dof_idx; // @todo
        adaflo_params.quad_index    = advec_diff_quad_idx;

        adaflo_params.concentration_subdivisions =
          base_in->parameters.advec_diff_adaflo_params.get_parameters().concentration_subdivisions;
        adaflo_params.convection_stabilization = false;
        adaflo_params.do_iteration             = false;
        adaflo_params.tol_nl_iteration =
          base_in->parameters.advec_diff_adaflo_params.get_parameters().tol_nl_iteration;

        adaflo_params.time.time_step_scheme =
          base_in->parameters.advec_diff_adaflo_params.get_parameters().time_step_scheme;
        adaflo_params.time.start_time =
          base_in->parameters.advec_diff_adaflo_params.get_parameters().start_time;
        adaflo_params.time.end_time =
          base_in->parameters.advec_diff_adaflo_params.get_parameters().end_time;
        adaflo_params.time.time_step_size_start =
          base_in->parameters.advec_diff_adaflo_params.get_parameters().time_step_size_start;
        adaflo_params.time.time_stepping_cfl =
          base_in->parameters.advec_diff_adaflo_params.get_parameters().time_stepping_cfl;
        adaflo_params.time.time_stepping_coef2 =
          base_in->parameters.advec_diff_adaflo_params.get_parameters().time_stepping_coef2;
        adaflo_params.time.time_step_tolerance =
          base_in->parameters.advec_diff_adaflo_params.get_parameters().time_step_tolerance;
        adaflo_params.time.time_step_size_max =
          base_in->parameters.advec_diff_adaflo_params.get_parameters().time_step_size_max;
        adaflo_params.time.time_step_size_min =
          base_in->parameters.advec_diff_adaflo_params.get_parameters().time_step_size_min;


        /*
         * Boundary conditions for the advected field
         */
        // @todo
        for (const auto &symmetry_id : base_in->get_symmetry_id("advection_diffusion"))
          bcs.symmetry.insert(symmetry_id);
        // for (const auto &dirichlet_bc : base_in->get_dirichlet_bc("advection_diffusion"))
        // bcs.dirichlet[dirichlet_bc.first] = dirichlet_bc.second;
        /*
         * initialize adaflo operation
         */
        advec_diff_operation = std::make_shared<LevelSetOKZSolverAdvanceConcentration<dim>>(
          advected_field,
          advected_field_old,
          advected_field_old_old,
          increment,
          rhs,
          velocity_vec_temp, //@todo: convert block vector to fe_system vector
          velocity_vec_temp, // only used if parameters.convection_stabilization = true
          velocity_vec_temp, // only used if parameters.convection_stabilization = true
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

        //@todo -- add
        // if (step_size_old > 0)
        // solution_update.sadd((step_size + step_size_old) / step_size_old,
        //-step_size / step_size_old,
        // solution_old);
        advec_diff_operation->advance_concentration(dt);
      }

      const LinearAlgebra::distributed::Vector<double> &
      get_advected_field() const
      {
        return advected_field;
      }

    private:
      void
      set_velocity(const LinearAlgebra::distributed::BlockVector<double> &vec)
      {
        VectorTools::convert_block_vector_to_fe_sytem_vector(
          vec,
          scratch_data.get_dof_handler(adaflo_params.dof_index_ls),
          velocity_vec_temp,
          scratch_data.get_dof_handler(adaflo_params.dof_index_vel));
      }

      void
      initialize_preconditioner()
      {
        // create diagonal preconditioner vector by assembly of mass matrix diagonal
        LinearAlgebra::distributed::Vector<double> diagonal(advected_field);
        diagonal = 1.0;
        preconditioner.reinit(diagonal);
      }

      const ScratchData<dim> &scratch_data;
      VectorType &            advected_field;
      BlockVectorType &       velocity_vec;
      /**
       * Boundary conditions for the advection diffusion operation
       * @todo
       */
      LevelSetOKZSolverAdvanceConcentrationBoundaryDescriptor<dim> bcs;

      LevelSetOKZSolverAdvanceConcentrationParameter adaflo_params;

      /**
       * Reference to the actual Navier-Stokes solver from adaflo
       */
      std::shared_ptr<LevelSetOKZSolverAdvanceConcentration<dim>> advec_diff_operation;

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
      VectorType                     velocity_vec_temp;
      /**
       *  advected field
       */

      VectorType advected_field_old;
      VectorType advected_field_old_old;
      VectorType increment;
      VectorType rhs;

      DoFHandler<dim>     dof_handler_adaflo;
      const FESystem<dim> fe_adaflo;
    };

    /**
     * Dummy specialization for 1. Needed to be able to compile
     * since the adaflo Navier-Stokes solver is not compiled for
     * 1D - due to the dependcy to parallel::distributed::Triangulation
     * and p4est.
     */
    template <>
    class AdafloWrapper<1>
    {
    private:
      using VectorType      = LinearAlgebra::distributed::Vector<double>;
      using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    public:
      /**
       * Constructor.
       */
      AdafloWrapper(ScratchData<1> &                   scratch_data,
                    const int                          advec_diff_dof_idx,
                    const int                          advec_diff_quad_idx,
                    const int                          velocity_dof_idx,
                    VectorType &                       solution_advected_field,
                    BlockVectorType &                  velocity_vec_in, // @todo: make const ref
                    std::shared_ptr<SimulationBase<1>> base_in)
      {
        (void)scratch_data;
        (void)advec_diff_dof_idx;
        (void)advec_diff_quad_idx;
        (void)velocity_dof_idx;
        (void)solution_advected_field;
        (void)velocity_vec_in;
        (void)base_in;

        AssertThrow(false, ExcNotImplemented());
      }
      void
      solve(const double dt)
      {
        (void)dt;
        AssertThrow(false, ExcNotImplemented());
      }

      const LinearAlgebra::distributed::Vector<double> &
      get_advected_field() const
      {
        AssertThrow(false, ExcNotImplemented());
      }

    private:
      void
      set_velocity(const LinearAlgebra::distributed::BlockVector<double> &vec)
      {
        (void)vec;
        AssertThrow(false, ExcNotImplemented());
      }

      void
      initialize_preconditioner()
      {
        AssertThrow(false, ExcNotImplemented());
      }
    };
  } // namespace AdvectionDiffusionAdaflo
} // namespace MeltPoolDG

#endif
