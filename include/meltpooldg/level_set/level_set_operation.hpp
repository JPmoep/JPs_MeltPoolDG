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
                 const unsigned int                             quad_idx_in)
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
        advec_diff_operation.initialize(
          scratch_data, solution_level_set_in, data_in, dof_idx, dof_no_bc_idx_in, quad_idx_in);
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
         *    initialize the curvature operation class
         */
        curvature_operation.initialize(scratch_data, data_in, dof_no_bc_idx_in, quad_idx_in);
        /*
         *    compute the curvature of the initial level set field
         */
        curvature_operation.solve(advec_diff_operation.solution_advected_field);
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
         *    compute the curvature
         */
        curvature_operation.solve(advec_diff_operation.solution_advected_field);
    }

    void
    compute_surface_tension(BlockVectorType & force_rhs, const double surface_tension_coefficient, const bool add = false)
    {
      
      scratch_data->get_matrix_free().template cell_loop<BlockVectorType, std::nullptr_t>(
        [&](const auto &matrix_free, auto &force_rhs, const auto &, auto macro_cells) {
            FECellIntegrator<dim, 1, double>   level_set(matrix_free,
                                                         dof_idx,
                                                         quad_idx);

            FECellIntegrator<dim, 1, double>   curvature(matrix_free, 
                                                         dof_no_bc_idx, 
                                                         quad_idx);

            FECellIntegrator<dim, dim, double> surface_tension(matrix_free,
                                                               dof_no_bc_idx,
                                                               quad_idx);

            for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
            {
              level_set.reinit(cell);
              level_set.gather_evaluate(solution_level_set, true, true);

              surface_tension.reinit(cell);

              curvature.reinit(cell);
              curvature.read_dof_values_plain(solution_curvature);
              curvature.evaluate(true, false);

              for (unsigned int q_index = 0; q_index < surface_tension.n_q_points; ++q_index)
              {
                  surface_tension.submit_value(surface_tension_coefficient * 
                                              level_set.get_gradient(q_index) *  // must be adopted --> level set be between zero and 1
                                              curvature.get_value(q_index), q_index); 
              }
              surface_tension.integrate_scatter(true, false, force_rhs);
            }
        },
        force_rhs,
        nullptr,
        !add);
    }
    /*
     *  getter functions for solution vectors
     */
    // @ todo

    private:
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
       *    This is the surface_tension vector calculated after level set and reinitialization update
       */
      //BlockVectorType surface_tension_force;


    };
  } // namespace LevelSet
} // namespace MeltPoolDG
