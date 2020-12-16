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
#include <meltpooldg/advection_diffusion/advection_diffusion_operation.hpp>
#include <meltpooldg/curvature/curvature_operation.hpp>
#include <meltpooldg/curvature/curvature_operation_adaflo_wrapper.hpp>
#include <meltpooldg/curvature/curvature_operation_base.hpp>
#include <meltpooldg/reinitialization/reinitialization_operation.hpp>
#include <meltpooldg/reinitialization/reinitialization_operation_adaflo_wrapper.hpp>
#include <meltpooldg/reinitialization/reinitialization_operation_base.hpp>

namespace MeltPoolDG
{
  namespace LevelSet
  {
    using namespace dealii;
    using namespace Reinitialization;
    using namespace AdvectionDiffusion;

    /*
     *     Level set model including advection, reinitialization and curvature computation
     *     of the level set function.
     */
    template <int dim>
    class LevelSetOperation
    {
    private:
      using VectorType      = LinearAlgebra::distributed::Vector<double>;
      using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    public:
      /*
       *  All the necessary parameters are stored in this vector.
       */
      LevelSetData<double> level_set_data;

      LevelSetOperation() = default;

      void
      initialize(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
                 const VectorType &                             solution_level_set_in,
                 const BlockVectorType &                        advection_velocity,
                 const Parameters<double> &                     data_in,
                 const unsigned int                             dof_idx_in,
                 const unsigned int                             dof_no_bc_idx_in,
                 const unsigned int                             quad_idx_in,
                 const unsigned int                             advection_dof_idx,
                 std::shared_ptr<SimulationBase<dim>>           base_in,
                 const unsigned int                             vel_dof_idx    = 0,
                 const unsigned int                             ls_zero_bc_idx = 0)
      {
        // parameters = data_in;
        scratch_data  = scratch_data_in;
        dof_idx       = dof_idx_in;
        dof_no_bc_idx = dof_no_bc_idx_in;
        quad_idx      = quad_idx_in;
        /*
         *  set the level set data
         */
        level_set_data = data_in.ls;
        /*
         *    initialize the advection diffusion operation and the reinitialization operation class
         */

        if ((data_in.advec_diff.implementation ==
             "meltpooldg")) // @todo: add stronger criterion for ls implementation == meltpooldg
          {
            advec_diff_operation =
              std::make_shared<AdvectionDiffusion::AdvectionDiffusionOperation<dim>>();
            advec_diff_operation->initialize(scratch_data,
                                             solution_level_set_in, // copy
                                             data_in,
                                             dof_idx,
                                             dof_no_bc_idx_in,
                                             quad_idx_in,
                                             advection_dof_idx);
          }
#ifdef MELT_POOL_DG_WITH_ADAFLO
        else if ((data_in.advec_diff.implementation == "adaflo") ||
                 (data_in.ls.implementation == "adaflo"))
          {
            advec_diff_operation =
              std::make_shared<AdvectionDiffusion::AdvectionDiffusionOperationAdaflo<dim>>(
                *scratch_data, ls_zero_bc_idx, quad_idx_in, vel_dof_idx, base_in, "level_set");

            advec_diff_operation->reinit();

            advec_diff_operation->set_initial_condition(solution_level_set_in, // copy
                                                        advection_velocity);
          }
#endif
        else
          AssertThrow(false, ExcNotImplemented());

        if ((data_in.reinit.implementation ==
             "meltpooldg")) // @todo: add stronger criterion for ls implementation == meltpooldg
          {
            reinit_operation = std::make_shared<Reinitialization::ReinitializationOperation<dim>>();
            reinit_operation->initialize(scratch_data,
                                         advec_diff_operation->get_advected_field(),
                                         data_in,
                                         dof_no_bc_idx_in,
                                         quad_idx_in);
          }
#ifdef MELT_POOL_DG_WITH_ADAFLO
        else if ((data_in.reinit.implementation == "adaflo") ||
                 (data_in.ls.implementation == "adaflo"))
          {
            AssertThrow(data_in.reinit.solver.do_matrix_free, ExcNotImplemented());
            reinit_operation =
              std::make_shared<Reinitialization::ReinitializationOperationAdaflo<dim>>(
                *scratch_data,
                dof_no_bc_idx_in,
                quad_idx_in,
                dof_no_bc_idx_in, // normal vec @todo
                advec_diff_operation->get_advected_field(),
                data_in);
          }
#endif
        else
          AssertThrow(false, ExcNotImplemented());
        /*
         *  set the parameters for the levelset problem; already determined parameters
         *  from the initialize call of advec_diff_operation are overwritten.
         */
        set_level_set_parameters(data_in);
        /*
         *  The initial solution of the level set equation will be reinitialized.
         */
        if (level_set_data.do_reinitialization)
          {
            while (!reinit_time_iterator.is_finished())
              {
                const double d_tau = reinit_time_iterator.get_next_time_increment();
                scratch_data->get_pcout() << std::setw(4) << ""
                                          << "| reini: τ= " << std::setw(10) << std::left
                                          << reinit_time_iterator.get_current_time();
                reinit_operation->solve(d_tau);
              }
            advec_diff_operation->get_advected_field() = reinit_operation->get_level_set();
            reinit_time_iterator.reset();
          }
        /*
         *    compute the smoothened function
         */
        transform_level_set_to_smooth_heaviside();
        /*
         *    initialize the curvature operation class
         */
        if ((data_in.curv.implementation ==
             "meltpooldg")) // @todo: add stronger criterion for ls implementation == meltpooldg
          {
            curvature_operation = std::make_shared<Curvature::CurvatureOperation<dim>>();

            curvature_operation->initialize(scratch_data, data_in, dof_no_bc_idx_in, quad_idx_in);
            /*
             *    compute the curvature of the initial level set field
             */
            curvature_operation->solve(advec_diff_operation->get_advected_field());
            /*
             *    correct the curvature value far away from the zero level set
             */
            if (level_set_data.do_curvature_correction)
              correct_curvature_values();
          }
#ifdef MELT_POOL_DG_WITH_ADAFLO
        else if ((data_in.curv.implementation == "adaflo") ||
                 (data_in.ls.implementation == "adaflo"))
          {
            AssertThrow(data_in.curv.do_matrix_free, ExcNotImplemented());
            curvature_operation = std::make_shared<Curvature::CurvatureOperationAdaflo<dim>>(
              *scratch_data_in,
              dof_no_bc_idx_in, //@todo
              dof_no_bc_idx_in, //@todo
              dof_no_bc_idx_in,
              quad_idx,
              advec_diff_operation->get_advected_field(),
              data_in);

            curvature_operation->solve(advec_diff_operation->get_advected_field());
          }
#endif
        else
          AssertThrow(false, ExcNotImplemented());

        // this->reinit();
      }

      void
      reinit()
      {
        advec_diff_operation->reinit();
        reinit_operation->reinit();
        curvature_operation->reinit();

        if (normal_vector_operation)
          normal_vector_operation->reinit();
      }

      void
      solve(const double dt, const BlockVectorType &advection_velocity)
      {
        /*
         *  solve the advection step of the level set function
         */
        advec_diff_operation->solve(dt, advection_velocity);
        /*
         *  solve the reinitialization problem of the level set equation
         */
        if (level_set_data.do_reinitialization)
          {
            reinit_operation->update_initial_solution(advec_diff_operation->get_advected_field());

            while (!reinit_time_iterator.is_finished())
              {
                const double d_tau = reinit_time_iterator.get_next_time_increment();
                scratch_data->get_pcout() << std::setw(4) << ""
                                          << "| reini: τ= " << std::setw(10) << std::left
                                          << reinit_time_iterator.get_current_time();
                reinit_operation->solve(d_tau);
                /*
                 *  reset the solution of the level set field to the reinitialized solution
                 */
                advec_diff_operation->get_advected_field() = reinit_operation->get_level_set();
              }


            reinit_time_iterator.reset();
          }
        /*
         *    compute the smoothened function
         */
        transform_level_set_to_smooth_heaviside();
        /*
         *    compute the curvature
         */
        curvature_operation->solve(advec_diff_operation->get_advected_field());
        /*
         *    correct the curvature value far away from the zero level set
         */
        if (level_set_data.do_curvature_correction)
          correct_curvature_values();
      }

      void
      compute_surface_tension(BlockVectorType &  force_rhs,
                              const double       surface_tension_coefficient,
                              const unsigned int flow_dof_idx,
                              const unsigned int flow_quad_idx,
                              const bool         zero_out = true)
      {
        scratch_data->get_matrix_free().template cell_loop<BlockVectorType, VectorType>(
          [&](const auto &matrix_free,
              auto &      force_rhs,
              const auto &level_set_as_heaviside,
              auto        macro_cells) {
            FECellIntegrator<dim, 1, double> level_set(matrix_free, dof_idx, flow_quad_idx);

            FECellIntegrator<dim, 1, double> curvature(matrix_free, dof_no_bc_idx, flow_quad_idx);

            FECellIntegrator<dim, dim, double> surface_tension(matrix_free,
                                                               flow_dof_idx,
                                                               flow_quad_idx);

            for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
              {
                level_set.reinit(cell);
                level_set.read_dof_values_plain(level_set_as_heaviside);
                level_set.evaluate(false, true);

                surface_tension.reinit(cell);

                curvature.reinit(cell);
                curvature.read_dof_values_plain(curvature_operation->get_curvature());
                curvature.evaluate(true, false);

                for (unsigned int q_index = 0; q_index < surface_tension.n_q_points; ++q_index)
                  {
                    surface_tension.submit_value(
                      surface_tension_coefficient *
                        level_set.get_gradient(
                          q_index) * // must be adopted --> level set be between zero and 1
                        curvature.get_value(q_index),
                      q_index);
                  }
                surface_tension.integrate_scatter(true, false, force_rhs);
              }
          },
          force_rhs,
          level_set_as_heaviside,
          zero_out);
      }
      /*
       *  getter functions for solution vectors
       */
      // @ todo

    private:
      inline double
      approximate_distance_from_level_set(const double phi, const double eps, const double cutoff)
      {
        if (std::abs(phi) < cutoff)
          return eps * std::log((1. + phi) / (1. - phi));
        else if (phi >= cutoff)
          return eps * std::log((1. + cutoff) / (1. - cutoff));
        else /*( phi <= -cutoff )*/
          return -eps * std::log((1. + cutoff) / (1. - cutoff));
      }

      /**
       * The given distance value is transformed to a smooth heaviside function \f$H_\epsilon\f$,
       * which has the property of \f$\int \nabla H_\epsilon=1\f$. This function has its transition
       * region between -2 and 2.
       */
      inline double
      smooth_heaviside_from_distance_value(const double x /*distance*/)
      {
        if (x > 0)
          return 1. - smooth_heaviside_from_distance_value(-x);
        else if (x < -2.)
          return 0;
        else if (x < -1.)
          {
            const double x2 = x * x;
            return (0.125 * (5. * x + x2) +
                    0.03125 * (-3. - 2. * x) * std::sqrt(-7. - 12. * x - 4. * x2) -
                    0.0625 * std::asin(std::sqrt(2.) * (x + 1.5)) + 23. * 0.03125 -
                    numbers::PI / 64.);
          }
        else
          {
            const double x2 = x * x;
            return (
              0.125 * (3. * x + x2) - 0.03125 * (-1. - 2. * x) * std::sqrt(1. - 4. * x - 4. * x2) +
              0.0625 * std::asin(std::sqrt(2.) * (x + 0.5)) + 15. * 0.03125 - numbers::PI / 64.);
          }
      }

      void
      transform_level_set_to_smooth_heaviside()
      {
        scratch_data->initialize_dof_vector(level_set_as_heaviside, dof_no_bc_idx);
        scratch_data->initialize_dof_vector(distance_to_level_set, dof_no_bc_idx);
        FEValues<dim> fe_values(scratch_data->get_mapping(),
                                scratch_data->get_dof_handler(this->dof_no_bc_idx).get_fe(),
                                scratch_data->get_quadrature(this->quad_idx),
                                update_values | update_quadrature_points | update_JxW_values);

        const unsigned int dofs_per_cell = scratch_data->get_n_dofs_per_cell();

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        const double cut_off_level_set = std::tanh(2);

        for (const auto &cell :
             scratch_data->get_dof_handler(this->dof_no_bc_idx).active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(local_dof_indices);

              const double epsilon_cell =
                reinit_constant_epsilon > 0.0 ?
                  reinit_constant_epsilon :
                  cell->diameter() / (std::sqrt(dim)) * reinit_scale_factor_epsilon;

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  const double distance = approximate_distance_from_level_set(
                    advec_diff_operation->get_advected_field()[local_dof_indices[i]],
                    epsilon_cell,
                    cut_off_level_set);
                  distance_to_level_set(local_dof_indices[i]) = distance;
                  level_set_as_heaviside(local_dof_indices[i]) =
                    smooth_heaviside_from_distance_value(2 * distance / (3 * epsilon_cell));
                }
            }
      }

      /// To avoid high-frequency errors in the curvature (spurious currents) the curvature is
      /// corrected to represent the value of the interface (zero level set). The approach by Zahedi
      /// et al. (2012) is pursued. Considering e.g. a bubble, the absolute curvature of areas
      /// outside of the bubble (Φ=-) must increase and vice-versa for areas
      ///  inside the bubble.
      //
      //           ******
      //       ****      ****
      //     **              **
      //    *      Φ=+         *  Φ=-
      //    *    sgn(d)=+      *  sgn(d)=-
      //    *                  *
      //     **              **
      //       ****      ****
      //           ******
      //
      void
      correct_curvature_values()
      {
        for (unsigned int i = 0; i < curvature_operation->get_curvature().local_size(); ++i)
          // if (std::abs(solution_curvature.local_element(i)) > 1e-4)
          if (1. - advec_diff_operation->get_advected_field().local_element(i) *
                     advec_diff_operation->get_advected_field().local_element(i) >
              1e-2)
            curvature_operation->get_curvature().local_element(i) =
              1. / (1. / curvature_operation->get_curvature().local_element(i) +
                    distance_to_level_set.local_element(i) / (dim - 1));
      }

      void
      set_level_set_parameters(const Parameters<double> &data_in)
      {
        level_set_data.do_reinitialization                = data_in.ls.do_reinitialization;
        advec_diff_operation->advec_diff_data.diffusivity = data_in.ls.artificial_diffusivity;
        advec_diff_operation->advec_diff_data.time_integration_scheme =
          data_in.ls.time_integration_scheme;
        advec_diff_operation->advec_diff_data.do_print_l2norm = data_in.ls.do_print_l2norm;
        advec_diff_operation->advec_diff_data.do_matrix_free  = data_in.ls.do_matrix_free;
        /*
         *  setup the time iterator for the reinitialization problem
         */
        reinit_time_iterator.initialize(
          TimeIteratorData<double>{0.0,
                                   100000.,
                                   data_in.reinit.dtau > 0.0 ?
                                     data_in.reinit.dtau :
                                     scratch_data->get_min_cell_size(dof_idx) *
                                       data_in.reinit.scale_factor_epsilon,
                                   data_in.reinit.max_n_steps,
                                   false});

        reinit_constant_epsilon     = data_in.reinit.constant_epsilon;     //@todo: better solution
        reinit_scale_factor_epsilon = data_in.reinit.scale_factor_epsilon; //@todo: better solution
      }

      std::shared_ptr<const ScratchData<dim>> scratch_data;
      /*
       *  The following objects are the operations, which are performed for solving the
       *  level set equation.
       */
      std::shared_ptr<AdvectionDiffusion::AdvectionDiffusionOperationBase<dim>>
                                                                            advec_diff_operation;
      std::shared_ptr<Reinitialization::ReinitializationOperationBase<dim>> reinit_operation;
      std::shared_ptr<Curvature::CurvatureOperationBase<dim>>               curvature_operation;
      /*
       *  The reinitialization of the level set function is a "pseudo"-time-dependent
       *  equation, which is solved up to quasi-steady state. Thus a time iterator is
       *  needed.
       */
      TimeIterator<double> reinit_time_iterator;
      /*
       * select the relevant DoFHandler
       */
      unsigned int dof_idx;
      unsigned int dof_no_bc_idx;
      unsigned int quad_idx;


      double reinit_constant_epsilon     = 0; //@todo: better solution
      double reinit_scale_factor_epsilon = 0; //@todo: better solution
    public:
      const LinearAlgebra::distributed::Vector<double> &
      get_curvature() const
      {
        return curvature_operation->get_curvature();
      }

      const LinearAlgebra::distributed::BlockVector<double> &
      get_normal_vector() const
      {
        return reinit_operation->get_normal_vector();
      }
      const LinearAlgebra::distributed::Vector<double> &
      get_level_set() const
      {
        return advec_diff_operation->get_advected_field();
      }

      LinearAlgebra::distributed::Vector<double> &
      get_level_set()
      {
        return advec_diff_operation->get_advected_field();
      }

      virtual void
      attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
      {
        advec_diff_operation->attach_vectors(vectors);
      }

      /*
       *   Computation of the normal vectors
       */
      std::shared_ptr<NormalVector::NormalVectorOperationBase<dim>> normal_vector_operation;
      /*
       *    This is the surface_tension vector calculated after level set and reinitialization
       * update
       */
      VectorType level_set_as_heaviside;
      VectorType distance_to_level_set;
    };
  } // namespace LevelSet
} // namespace MeltPoolDG
