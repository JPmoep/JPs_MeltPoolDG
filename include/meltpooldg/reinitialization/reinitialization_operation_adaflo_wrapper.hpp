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
#  include <meltpooldg/reinitialization/reinitialization_operation_base.hpp>
#  include <meltpooldg/utilities/vector_tools.hpp>

#  include <adaflo/diagonal_preconditioner.h>
#  include <adaflo/level_set_okz_compute_normal.h>
#  include <adaflo/level_set_okz_preconditioner.h>
#  include <adaflo/level_set_okz_reinitialization.h>

namespace MeltPoolDG
{
  namespace Reinitialization
  {
    template <int dim>
    class ReinitializationOperationAdaflo
      : public MeltPoolDG::Reinitialization::ReinitializationOperationBase<dim>
    {
    private:
      using VectorType      = LinearAlgebra::distributed::Vector<double>;
      using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    public:
      /**
       * Constructor.
       */
      ReinitializationOperationAdaflo(ScratchData<dim> &scratch_data,
                                      const int         reinit_dof_idx,
                                      const int         reinit_quad_idx,
                                      const int         normal_dof_idx,
                                      const VectorType &initial_solution_level_set,
                                      std::shared_ptr<SimulationBase<dim>> base_in)
        : scratch_data(scratch_data)
        , reinit_dof_idx(reinit_dof_idx)
        , reinit_quad_idx(reinit_quad_idx)
      {
        /**
         * set parameters of adaflo
         */
        set_adaflo_parameters(base_in->parameters, reinit_dof_idx, reinit_quad_idx, normal_dof_idx);

        /*
         * initialize normal_vector_operation from adaflo
         */
        normal_vector_operation_adaflo =
          std::make_shared<NormalVector::NormalVectorOperationAdaflo<dim>>(scratch_data,
                                                                           reinit_dof_idx,
                                                                           normal_dof_idx,
                                                                           reinit_quad_idx,
                                                                           level_set,
                                                                           base_in->parameters);

        compute_normal = [&](bool do_compute_normal) {
          if (do_compute_normal && force_compute_normal)
            normal_vector_operation_adaflo->solve(level_set);
        };

        /*
         * initialize reinitialization operation from adaflo
         */
        reinit_operation_adaflo = std::make_shared<LevelSetOKZSolverReinitialization<dim>>(
          normal_vector_operation_adaflo->get_solution_normal_vector(),
          scratch_data.get_cell_diameters(),
          cell_diameter_max,
          cell_diameter_min,
          scratch_data.get_constraint(reinit_dof_idx),
          increment,
          level_set,
          rhs,
          scratch_data.get_pcout(reinit_dof_idx),
          preconditioner,
          last_concentration_range, // @todo
          reinit_params_adaflo,
          first_reinit_step,
          scratch_data.get_matrix_free());

        /**
         *  initialize the dof vectors
         */
        reinit();

        /**
         *  set initial solution of level set
         */

        level_set.copy_locally_owned_data_from(initial_solution_level_set);
      }

      void
      reinit() override
      {
        initialize_vectors();

        compute_cell_diameters<dim>(scratch_data.get_matrix_free(),
                                    reinit_dof_idx,
                                    cell_diameters,
                                    cell_diameter_min,
                                    cell_diameter_max);

        /**
         * initialize the preconditioner
         */
        initialize_mass_matrix_diagonal<dim, double>(scratch_data.get_matrix_free(),
                                                     scratch_data.get_constraint(reinit_dof_idx),
                                                     reinit_dof_idx,
                                                     reinit_quad_idx,
                                                     preconditioner);

        normal_vector_operation_adaflo->reinit();
      }

      /**
       * Solver time step
       */
      void
      solve(const double dt) override
      {
        reinit_operation_adaflo->reinitialize(dt,
                                              1 /*stab_steps @todo*/,
                                              0 /*diff_steps @todo*/,
                                              compute_normal);
        scratch_data.get_pcout() << "\t |ΔΨ|∞ = " << std::setw(15) << std::left
                                 << std::setprecision(10) << increment.linfty_norm();
        scratch_data.get_pcout() << " |ΔΨ|²/dT = " << std::setw(15) << std::left
                                 << std::setprecision(10) << increment.l2_norm() / dt << "|"
                                 << std::endl;
        force_compute_normal = false;
      }

      const LinearAlgebra::distributed::Vector<double> &
      get_level_set() const override
      {
        return level_set;
      }

      LinearAlgebra::distributed::Vector<double> &
      get_level_set() override
      {
        return level_set;
      }

      const LinearAlgebra::distributed::BlockVector<double> &
      get_normal_vector() const override
      {
        return normal_vector_operation_adaflo->get_solution_normal_vector();
      }

      void
      attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) override
      {
        vectors.push_back(&level_set);
      }

      void
      update_initial_solution(const VectorType &level_set_in) override
      {
        (void)level_set_in;
        force_compute_normal = true;
      }

    private:
      void
      set_adaflo_parameters(const Parameters<double> &parameters,
                            const int                 reinit_dof_idx,
                            const int                 reinit_quad_idx,
                            const int                 normal_dof_idx)
      {
        reinit_params_adaflo.time.start_time           = 0.0;
        reinit_params_adaflo.time.end_time             = 1e8;
        reinit_params_adaflo.time.time_step_size_start = parameters.reinit.dtau;
        reinit_params_adaflo.time.time_step_size_min   = parameters.reinit.dtau;
        reinit_params_adaflo.time.time_step_size_max   = parameters.reinit.dtau;

        //@todo?
        // if (parameters.reinit.time_integration_scheme == "implicit_euler")
        // reinit_params_adaflo.time.time_step_scheme =
        // TimeSteppingParameters::Scheme::implicit_euler;
        //
        reinit_params_adaflo.time.time_stepping_cfl   = 0.8;  //@ todo
        reinit_params_adaflo.time.time_stepping_coef2 = 10.0; //@ todo capillary number

        reinit_params_adaflo.dof_index_ls     = reinit_dof_idx;
        reinit_params_adaflo.dof_index_normal = normal_dof_idx;
        reinit_params_adaflo.quad_index       = reinit_quad_idx;
        reinit_params_adaflo.do_iteration     = false; //@ todo
      }

      void
      initialize_vectors()
      {
        /**
         * initialize advected field dof vectors
         */
        scratch_data.initialize_dof_vector(level_set, reinit_params_adaflo.dof_index_ls);
        /**
         * initialize vectors for the solution of the linear system
         */
        scratch_data.initialize_dof_vector(rhs, reinit_params_adaflo.dof_index_ls);
        scratch_data.initialize_dof_vector(increment, reinit_params_adaflo.dof_index_ls);
      }

    private:
      const ScratchData<dim> &scratch_data;

      const unsigned int reinit_dof_idx;
      const unsigned int reinit_quad_idx;
      /**
       *  advected field
       */
      VectorType level_set;
      /**
       *  vectors for the solution of the linear system
       */
      VectorType increment;
      VectorType rhs;

      /**
       * Adaflo parameters for the level set problem
       */
      LevelSetOKZSolverReinitializationParameter reinit_params_adaflo;

      /**
       * Reference to the actual advection diffusion solver from adaflo
       */
      std::shared_ptr<NormalVector::NormalVectorOperationAdaflo<dim>>
                                                              normal_vector_operation_adaflo;
      std::shared_ptr<LevelSetOKZSolverReinitialization<dim>> reinit_operation_adaflo;

      /**
       *  Diagonal preconditioner @todo
       */
      DiagonalPreconditioner<double> preconditioner;

      std::pair<double, double>              last_concentration_range;
      AlignedVector<VectorizedArray<double>> cell_diameters;
      double                                 cell_diameter_min;
      double                                 cell_diameter_max;
      bool                                   first_reinit_step;
      bool                                   force_compute_normal = true;
      std::function<void(bool)>              compute_normal;
    };
  } // namespace Reinitialization
} // namespace MeltPoolDG

#endif
