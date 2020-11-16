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
#include <meltpooldg/reinitialization/reinitialization_operation.hpp>

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
                 const Parameters<double> &                     data_in,
                 const unsigned int                             dof_idx_in,
                 const unsigned int                             dof_no_bc_idx_in,
                 const unsigned int                             quad_idx_in,
                 const unsigned int                             advection_dof_idx)
      {
        scratch_data  = scratch_data_in;
        dof_idx       = dof_idx_in;
        dof_no_bc_idx = dof_no_bc_idx_in;
        quad_idx      = quad_idx_in;
        /*
         *  set the level set data
         */
        level_set_data = data_in.ls;
        /*
         *  initialize the advection_diffusion problem
         */
        advec_diff_operation.initialize(scratch_data,
                                        solution_level_set_in,
                                        data_in,
                                        dof_idx,
                                        dof_no_bc_idx_in,
                                        quad_idx_in,
                                        advection_dof_idx);
        /*
         *  set the parameters for the levelset problem; already determined parameters
         *  from the initialize call of advec_diff_operation are overwritten.
         */
        set_level_set_parameters(data_in);
        /*
         *    initialize the reinitialization operation class
         */
        reinit_operation.initialize(
          scratch_data, solution_level_set_in, data_in, dof_no_bc_idx_in, quad_idx_in);
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
                reinit_operation.solve(d_tau);
              }
            advec_diff_operation.solution_advected_field =
              reinit_operation.solution_level_set; // @ could be defined by reference
            reinit_time_iterator.reset();
          }
        /*
         *    compute the smoothened function
         */
        transform_level_set_to_smooth_heaviside();
        /*
         *    initialize the curvature operation class
         */
        curvature_operation.initialize(scratch_data, data_in, dof_no_bc_idx_in, quad_idx_in);
        /*
         *    compute the curvature of the initial level set field
         */
        curvature_operation.solve(advec_diff_operation.solution_advected_field);
        /*
         *    correct the curvature value far away from the zero level set
         */
        if (level_set_data.do_curvature_correction)
          correct_curvature_values();
      }

      void
      solve(const double dt, const BlockVectorType &advection_velocity)
      {
        /*
         *  solve the advection step of the level set function
         */
        advec_diff_operation.solve(dt, advection_velocity);
        /*
         *  solve the reinitialization problem of the level set equation
         */
        if (level_set_data.do_reinitialization)
          {
            reinit_operation.update_initial_solution(advec_diff_operation.solution_advected_field);

            while (!reinit_time_iterator.is_finished())
              {
                const double d_tau = reinit_time_iterator.get_next_time_increment();
                scratch_data->get_pcout() << std::setw(4) << ""
                                          << "| reini: τ= " << std::setw(10) << std::left
                                          << reinit_time_iterator.get_current_time();
                reinit_operation.solve(d_tau);
              }

            /*
             *  reset the solution of the level set field to the reinitialized solution
             */
            advec_diff_operation.solution_advected_field =
              reinit_operation.solution_level_set; // @ could be defined by reference
            reinit_time_iterator.reset();
          }
        /*
         *    compute the smoothened function
         */
        transform_level_set_to_smooth_heaviside();
        /*
         *    compute the curvature
         */
        curvature_operation.solve(advec_diff_operation.solution_advected_field);
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
        level_set_as_heaviside.update_ghost_values();
        scratch_data->get_matrix_free().template cell_loop<BlockVectorType, std::nullptr_t>(
          [&](const auto &matrix_free, auto &force_rhs, const auto &, auto macro_cells) {
            FECellIntegrator<dim, 1, double> level_set(matrix_free, dof_idx, flow_quad_idx);

            FECellIntegrator<dim, 1, double> curvature(matrix_free, dof_no_bc_idx, flow_quad_idx);

            FECellIntegrator<dim, dim, double> surface_tension(matrix_free,
                                                               flow_dof_idx,
                                                               flow_quad_idx);

            for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
              {
                level_set.reinit(cell);
                level_set.gather_evaluate(level_set_as_heaviside, false, true);

                surface_tension.reinit(cell);

                curvature.reinit(cell);
                curvature.read_dof_values_plain(solution_curvature);
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
          nullptr,
          zero_out);
        level_set_as_heaviside.zero_out_ghosts();
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

              const double epsilon_cell = reinit_operation.reinit_data.constant_epsilon > 0.0 ?
                                            reinit_operation.reinit_data.constant_epsilon :
                                            cell->diameter() / (std::sqrt(dim)) *
                                              reinit_operation.reinit_data.scale_factor_epsilon;

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  const double distance =
                    approximate_distance_from_level_set(solution_level_set[local_dof_indices[i]],
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
        for (unsigned int i = 0; i < solution_curvature.local_size(); ++i)
          // if (std::abs(solution_curvature.local_element(i)) > 1e-4)
          if (1. - solution_level_set.local_element(i) * solution_level_set.local_element(i) > 1e-2)
            curvature_operation.solution_curvature.local_element(i) =
              1. / (1. / curvature_operation.solution_curvature.local_element(i) +
                    distance_to_level_set.local_element(i) / (dim - 1));
      }

      void
      set_level_set_parameters(const Parameters<double> &data_in)
      {
        level_set_data.do_reinitialization                   = data_in.ls.do_reinitialization;
        advec_diff_operation.advec_diff_data.diffusivity     = data_in.ls.artificial_diffusivity;
        advec_diff_operation.advec_diff_data.theta           = data_in.ls.theta;
        advec_diff_operation.advec_diff_data.do_print_l2norm = data_in.ls.do_print_l2norm;
        advec_diff_operation.advec_diff_data.do_matrix_free  = data_in.ls.do_matrix_free;
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
      }

      std::shared_ptr<const ScratchData<dim>> scratch_data;
      /*
       *  The following objects are the operations, which are performed for solving the
       *  level set equation.
       */
      AdvectionDiffusionOperation<dim>   advec_diff_operation;
      ReinitializationOperation<dim>     reinit_operation;
      Curvature::CurvatureOperation<dim> curvature_operation;

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

    public:
      /*
       *    This is the primary solution variable of this module, which will be also publically
       *    accessible for output_results.
       */
      const VectorType &solution_level_set = advec_diff_operation.solution_advected_field;
      /*
       *    This is the curvature solution variable, which will be publically
       *    accessible for output_results.
       */
      const VectorType &solution_curvature = curvature_operation.solution_curvature;
      /*
       *    This is the normal vector field, which will be publically
       *    accessible for output_results.
       */
      const BlockVectorType &solution_normal_vector = reinit_operation.solution_normal_vector;
      /*
       *    This is the surface_tension vector calculated after level set and reinitialization
       * update
       */
      VectorType level_set_as_heaviside;
      VectorType distance_to_level_set;
    };
  } // namespace LevelSet
} // namespace MeltPoolDG
