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
    template <int dim, int comp_advec_diff = 0, int comp_no_dirichlet = 1>
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
                 const Parameters<double> &                     data_in)
      {
        scratch_data = scratch_data_in;
        /*
         *  set the level set data
         */
        level_set_data = data_in.ls;
        /*
         *  initialize the advection_diffusion problem
         */
        advec_diff_operation.initialize(scratch_data, solution_level_set_in, data_in);
        /*
         *  set the parameters for the levelset problem; already determined parameters
         *  from the initialize call of advec_diff_operation are overwritten.
         */
        set_level_set_parameters(data_in);
        /*
         *    initialize the reinitialization operation class
         */
        reinit_operation.initialize(scratch_data, solution_level_set_in, data_in);
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
        curvature_operation.initialize(scratch_data, data_in);
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
                                     scratch_data->get_min_cell_size() *
                                       data_in.reinit.scale_factor_epsilon,
                                   data_in.reinit.max_n_steps,
                                   false});
      }

      std::shared_ptr<const ScratchData<dim>> scratch_data;
      /*
       *  The following objects are the operations, which are performed for solving the
       *  level set equation.
       */
      AdvectionDiffusionOperation<dim, comp_advec_diff, comp_no_dirichlet> advec_diff_operation;
      ReinitializationOperation<dim, comp_no_dirichlet>                    reinit_operation;
      Curvature::CurvatureOperation<dim, comp_no_dirichlet>                curvature_operation;

      /*
       *  The reinitialization of the level set function is a "pseudo"-time-dependent
       *  equation, which is solved up to quasi-steady state. Thus a time iterator is
       *  needed.
       */
      TimeIterator<double> reinit_time_iterator;

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
    };
  } // namespace LevelSet
} // namespace MeltPoolDG
