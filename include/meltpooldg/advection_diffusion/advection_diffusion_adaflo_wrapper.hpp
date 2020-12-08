/* ---------------------------------------------------------------------
 *
 * Author: Peter Münch, Magdalena Schreter, TUM, December 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

#ifdef MELT_POOL_DG_WITH_ADAFLO

#  include <deal.II/lac/generic_linear_algebra.h>

#  include <meltpooldg/interface/scratch_data.hpp>
#  include <meltpooldg/interface/simulationbase.hpp>
#  include <meltpooldg/utilities/vector_tools.hpp>

#  include <adaflo/diagonal_preconditioner.h>
#  include <adaflo/level_set_okz_advance_concentration.h>

namespace MeltPoolDG
{
  namespace AdvectionDiffusionAdaflo
  {
    template <int dim>
    class AdafloWrapper
      : public MeltPoolDG::AdvectionDiffusion::AdvectionDiffusionOperationBase<dim>
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
                    const VectorType                     initial_solution_advected_field,
                    const BlockVectorType                velocity_vec_in, // @todo: make const ref
                    std::shared_ptr<SimulationBase<dim>> base_in)
        : scratch_data(scratch_data)
      {
        /**
         * set parameters of adaflo
         */
        set_adaflo_parameters(base_in->parameters,
                              advec_diff_dof_idx,
                              advec_diff_quad_idx,
                              velocity_dof_idx);
        /**
         *  initialize the dof vectors
         */
        initialize_vectors();
        /**
         *  set initial solution of advected field
         */
        advected_field.copy_locally_owned_data_from(initial_solution_advected_field);
        advected_field_old     = advected_field;
        advected_field_old_old = advected_field;

        /**
         * initialize the preconditioner
         */
        initialize_preconditioner();
        /**
         *  set velocity
         */
        set_velocity(velocity_vec_in);
        /*
         * Boundary conditions for the advected field
         */
        for (const auto &symmetry_id : base_in->get_symmetry_id("advection_diffusion"))
          bcs.symmetry.insert(symmetry_id);
        for (const auto &dirichlet_bc : base_in->get_dirichlet_bc("advection_diffusion"))
          bcs.dirichlet[dirichlet_bc.first] = dirichlet_bc.second;
        /*
         * initialize adaflo operation
         */
        advec_diff_operation = std::make_shared<LevelSetOKZSolverAdvanceConcentration<dim>>(
          advected_field,
          advected_field_old,
          advected_field_old_old,
          increment,
          rhs,
          velocity_vec,
          velocity_vec_old,
          velocity_vec_old_old,
          scratch_data.get_diameter(advec_diff_dof_idx),
          scratch_data.get_cell_diameters(advec_diff_dof_idx),
          scratch_data.get_constraint(advec_diff_dof_idx),
          scratch_data.get_pcout(advec_diff_dof_idx),
          bcs,
          scratch_data.get_matrix_free(),
          adaflo_params,
          global_max_velocity,
          preconditioner);
      }

      /**
       * Solver time step
       */
      void
      solve(const double dt, const BlockVectorType &current_velocity) override
      {
        advected_field_old_old.copy_locally_owned_data_from(advected_field_old);
        advected_field_old.copy_locally_owned_data_from(advected_field);

        advected_field.update_ghost_values();
        advected_field_old.update_ghost_values();
        advected_field_old_old.update_ghost_values();

        set_velocity(current_velocity);

        //@todo -- extrapolation (?)
        // if (step_size_old > 0)
        // solution_update.sadd((step_size + step_size_old) / step_size_old,
        //-step_size / step_size_old,
        // solution_old);
        advec_diff_operation->advance_concentration(dt);

        scratch_data.get_pcout() << " |phi|= " << std::setw(15) << std::setprecision(10)
                                 << std::left << get_advected_field().l2_norm()
                                 << " |phi_n-1|= " << std::setw(15) << std::setprecision(10)
                                 << std::left << get_advected_field_old().l2_norm()
                                 << " |phi_n-2|= " << std::setw(15) << std::setprecision(10)
                                 << std::left << get_advected_field_old_old().l2_norm()
                                 << std::endl;
      }

      const LinearAlgebra::distributed::Vector<double> &
      get_advected_field() const
      {
        return advected_field;
      }

      const LinearAlgebra::distributed::Vector<double> &
      get_advected_field_old() const
      {
        return advected_field_old;
      }

      const LinearAlgebra::distributed::Vector<double> &
      get_advected_field_old_old() const
      {
        return advected_field_old_old;
      }

    private:
      void
      set_adaflo_parameters(const Parameters<double> &parameters,
                            const int                 advec_diff_dof_idx,
                            const int                 advec_diff_quad_idx,
                            const int                 velocity_dof_idx)
      {
        adaflo_params.time.start_time           = parameters.advec_diff.start_time;
        adaflo_params.time.end_time             = parameters.advec_diff.end_time;
        adaflo_params.time.time_step_size_start = parameters.advec_diff.time_step_size;
        adaflo_params.time.time_step_size_min   = parameters.advec_diff.time_step_size;
        adaflo_params.time.time_step_size_max   = parameters.advec_diff.time_step_size;
        adaflo_params.time.time_step_scheme =
          (parameters.advec_diff.theta == 1.0) ? TimeSteppingParameters::Scheme::implicit_euler :
          (parameters.advec_diff.theta == 0.0) ? TimeSteppingParameters::Scheme::explicit_euler :
          (parameters.advec_diff.theta == 0.5) ? TimeSteppingParameters::Scheme::crank_nicolson :
                                                 TimeSteppingParameters::Scheme::bdf_2;

        /**
         *  set number of concentration subdivisions equal to degree
         */
        adaflo_params.concentration_subdivisions = parameters.base.degree;

        adaflo_params.dof_index_ls  = advec_diff_dof_idx;
        adaflo_params.dof_index_vel = velocity_dof_idx;
        adaflo_params.quad_index    = advec_diff_quad_idx;

        adaflo_params.convection_stabilization = false; //@ todo
        adaflo_params.do_iteration             = false; //@ todo
        adaflo_params.tol_nl_iteration         = false; //@ todo

        adaflo_params.time.time_stepping_cfl   = 0.8;  //@ todo
        adaflo_params.time.time_stepping_coef2 = 10.0; //@ todo capillary number
      }

      void
      set_velocity(const LinearAlgebra::distributed::BlockVector<double> &vec)
      {
        velocity_vec_old_old.zero_out_ghosts();
        velocity_vec_old.zero_out_ghosts();
        velocity_vec.zero_out_ghosts();

        velocity_vec_old_old = velocity_vec_old;
        velocity_vec_old     = velocity_vec;

        VectorTools::convert_block_vector_to_fe_sytem_vector(
          vec,
          scratch_data.get_dof_handler(adaflo_params.dof_index_ls),
          velocity_vec,
          scratch_data.get_dof_handler(adaflo_params.dof_index_vel));

        velocity_vec_old_old.update_ghost_values();
        velocity_vec_old.update_ghost_values();
        velocity_vec.update_ghost_values();
      }

      void
      initialize_vectors()
      {
        /**
         * initialize advected field dof vectors
         */
        scratch_data.initialize_dof_vector(advected_field, adaflo_params.dof_index_ls);
        scratch_data.initialize_dof_vector(advected_field_old, adaflo_params.dof_index_ls);
        scratch_data.initialize_dof_vector(advected_field_old_old, adaflo_params.dof_index_ls);
        /**
         * initialize vectors for the solution of the linear system
         */
        scratch_data.initialize_dof_vector(rhs, adaflo_params.dof_index_ls);
        scratch_data.initialize_dof_vector(increment, adaflo_params.dof_index_ls);
        /**
         *  initialize the velocity vector for adaflo
         */
        scratch_data.initialize_dof_vector(velocity_vec, adaflo_params.dof_index_vel);
        scratch_data.initialize_dof_vector(velocity_vec_old, adaflo_params.dof_index_vel);
        scratch_data.initialize_dof_vector(velocity_vec_old_old, adaflo_params.dof_index_vel);
      }

      void
      initialize_preconditioner()
      {
        // create diagonal preconditioner vector by assembly of mass matrix diagonal
        LinearAlgebra::distributed::Vector<double> diagonal(advected_field);
        diagonal = 1.0;
        preconditioner.reinit(diagonal);
      }

    private:
      const ScratchData<dim> &scratch_data;
      /**
       *  advected field
       */
      VectorType advected_field;
      VectorType advected_field_old;
      VectorType advected_field_old_old;
      /**
       *  vectors for the solution of the linear system
       */
      VectorType increment;
      VectorType rhs;

      /**
       *  velocity
       */
      VectorType velocity_vec;
      VectorType velocity_vec_old;
      VectorType velocity_vec_old_old;
      /**
       * Boundary conditions for the advection diffusion operation
       */
      LevelSetOKZSolverAdvanceConcentrationBoundaryDescriptor<dim> bcs;
      /**
       * Adaflo parameters for the level set problem
       */
      LevelSetOKZSolverAdvanceConcentrationParameter adaflo_params;

      /**
       * Reference to the actual advection diffusion solver from adaflo
       */
      std::shared_ptr<LevelSetOKZSolverAdvanceConcentration<dim>> advec_diff_operation;

      /**
       *  maximum velocity --> set by adaflo
       */
      double global_max_velocity;
      /**
       *  Diagonal preconditioner @todo
       */
      DiagonalPreconditioner<double> preconditioner;
    };

    /**
     * Dummy specialization for 1. Needed to be able to compile
     * since the adaflo Navier-Stokes solver is not compiled for
     * 1D - due to the dependcy to parallel::distributed::Triangulation
     * and p4est.
     */
    template <>
    class AdafloWrapper<1>
      : public MeltPoolDG::AdvectionDiffusion::AdvectionDiffusionOperationBase<1>
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
      solve(const double dt, const BlockVectorType &vec)
      {
        (void)dt;
        (void)vec;
        AssertThrow(false, ExcNotImplemented());
      }

      const LinearAlgebra::distributed::Vector<double> &
      get_advected_field() const
      {
        AssertThrow(false, ExcNotImplemented());
      }
      const LinearAlgebra::distributed::Vector<double> &
      get_advected_field_old() const
      {
        AssertThrow(false, ExcNotImplemented());
      }
      const LinearAlgebra::distributed::Vector<double> &
      get_advected_field_old_old() const
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
