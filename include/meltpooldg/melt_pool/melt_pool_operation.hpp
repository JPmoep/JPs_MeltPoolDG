/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, November 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
// DoFTools
#include <deal.II/dofs/dof_tools.h>
// MeltPoolDG
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/utilities/utilityfunctions.hpp>

namespace MeltPoolDG
{
  namespace MeltPool
  {
    using namespace dealii;

    template <int dim>
    class MeltPoolOperation
    {
    private:
      using VectorType      = LinearAlgebra::distributed::Vector<double>;
      using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    public:
      /*
       *    This are the primary solution variables of this module, which will be also publically
       *    accessible for output_results.
       */
      VectorType temperature;
      VectorType solid;
      VectorType liquid;
      /*
       *  All the necessary parameters are stored in this struct.
       */
      MeltPoolData<double> mp_data;
      FlowData<double>     flow_data;

      MeltPoolOperation() = default;

      void
      initialize(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
                 const Parameters<double> &                     data_in,
                 const unsigned int                             ls_dof_idx_in,
                 const unsigned int                             flow_vel_dof_idx_in,
                 const unsigned int                             flow_quad_idx_in,
                 const unsigned int                             temp_dof_idx_in,
                 const unsigned int                             temp_quad_idx_in,
                 const VectorType &                             level_set_as_heaviside,
                 const double                                   start_time_in)
      {
        scratch_data     = scratch_data_in;
        ls_dof_idx       = ls_dof_idx_in;
        flow_vel_dof_idx = flow_vel_dof_idx_in;
        flow_quad_idx    = flow_quad_idx_in;
        temp_dof_idx     = temp_dof_idx_in;
        temp_quad_idx    = temp_quad_idx_in;
        time             = start_time_in;
        /*
         *  set the advection diffusion data
         */
        mp_data = data_in.mp;
        /*
         *  set the parameters for the melt pool operation
         */
        set_melt_pool_parameters(data_in);
        /*
         *  Get the center point of the laser source
         */
        laser_center =
          MeltPoolDG::UtilityFunctions::convert_string_coords_to_point<dim>(mp_data.laser_center);
        /*
         *  Initialize the temperature field
         */
        compute_temperature_vector(level_set_as_heaviside);
      }

      /**
       * The force contribution of the recoil pressure due to evaporation is computed. The model of
       * S.I. Anisimov and V.A. Khokhlov (1995) is considered. The consideration of any other model
       * is however possible. First, the temperature is updated and second, the recoil pressure is
       * computed.
       */
      void
      compute_recoil_pressure_force(VectorType &      force_rhs,
                                    const VectorType &level_set_as_heaviside,
                                    const double      dt,
                                    bool              zero_out = true)
      {
        time += dt;
        // 1) compute the current center of the laser beam
        if (mp_data.do_move_laser)
          laser_center[0] += mp_data.scan_speed * dt;

        // 2) update the temperature field
        compute_temperature_vector(level_set_as_heaviside);

        temperature.update_ghost_values();
        // 3) update the recoil pressure force
        scratch_data->get_matrix_free().template cell_loop<VectorType, VectorType>(
          [&](const auto &matrix_free,
              auto &      force_rhs,
              const auto &level_set_as_heaviside,
              auto        macro_cells) {
            FECellIntegrator<dim, 1, double>   level_set(matrix_free, ls_dof_idx, flow_quad_idx);
            FECellIntegrator<dim, dim, double> recoil_pressure(matrix_free,
                                                               flow_vel_dof_idx,
                                                               flow_quad_idx);

            FECellIntegrator<dim, 1, double> temperature_val(matrix_free,
                                                             temp_dof_idx,
                                                             flow_quad_idx);

            for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
              {
                level_set.reinit(cell);
                level_set.gather_evaluate(level_set_as_heaviside, false, true);

                temperature_val.reinit(cell);
                temperature_val.read_dof_values_plain(temperature);
                temperature_val.evaluate(true, false);

                recoil_pressure.reinit(cell);

                for (unsigned int q_index = 0; q_index < recoil_pressure.n_q_points; ++q_index)
                  {
                    const auto &t = temperature_val.get_value(q_index);

                    VectorizedArray<double> recoil_pressure_coefficient = 0;

                    for (unsigned int v = 0; v < VectorizedArray<double>::size(); ++v)
                      recoil_pressure_coefficient[v] = compute_recoil_pressure_coefficient(t[v]);

                    recoil_pressure.submit_value(recoil_pressure_coefficient *
                                                   level_set.get_gradient(q_index),
                                                 q_index);
                  }
                recoil_pressure.integrate_scatter(true, false, force_rhs);
              }
          },
          force_rhs,
          level_set_as_heaviside,
          zero_out);
        temperature.zero_out_ghosts();
      }

      /**
       *  This function introduces the basic framework for temperature-dependent surface tension
       *  forces, i.e. Marangoni convection.
       */
      void
      compute_temperature_dependent_surface_tension(
        VectorType &       force_rhs,
        const VectorType & level_set_as_heaviside,
        const VectorType & solution_curvature,
        const double       surface_tension_coefficient,
        const double       temperature_dependent_surface_tension_coefficient,
        const double       surface_tension_reference_temperature,
        const unsigned int ls_dof_idx,
        const unsigned int flow_vel_dof_idx,
        const unsigned int flow_quad_idx,
        const unsigned int temp_dof_idx,
        const bool         zero_out = true)
      {
        solution_curvature.update_ghost_values();

        scratch_data->get_matrix_free().template cell_loop<VectorType, VectorType>(
          [&](const auto &matrix_free,
              auto &      force_rhs,
              const auto &level_set_as_heaviside,
              auto        macro_cells) {
            FECellIntegrator<dim, 1, double> level_set(matrix_free, ls_dof_idx, flow_quad_idx);

            FECellIntegrator<dim, 1, double> curvature(
              matrix_free, temp_dof_idx, flow_quad_idx); /*@todo: own index for curvature*/

            FECellIntegrator<dim, 1, double> temperature_val(matrix_free,
                                                             temp_dof_idx,
                                                             flow_quad_idx);

            FECellIntegrator<dim, dim, double> surface_tension(matrix_free,
                                                               flow_vel_dof_idx,
                                                               flow_quad_idx);

            const double &alpha0   = surface_tension_coefficient;
            const double &d_alpha0 = temperature_dependent_surface_tension_coefficient;
            const auto    T0       = VectorizedArray<double>(surface_tension_reference_temperature);

            for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
              {
                level_set.reinit(cell);
                level_set.gather_evaluate(level_set_as_heaviside, false, true);

                surface_tension.reinit(cell);

                curvature.reinit(cell);
                curvature.read_dof_values_plain(solution_curvature);
                curvature.evaluate(true, false);

                temperature_val.reinit(cell);
                temperature_val.read_dof_values_plain(temperature);
                temperature_val.evaluate(true, true);

                for (unsigned int q_index = 0; q_index < surface_tension.n_q_points; ++q_index)
                  {
                    const auto n      = level_set.get_gradient(q_index);
                    const auto T      = temperature_val.get_value(q_index);
                    const auto grad_T = temperature_val.get_gradient(q_index);

                    Tensor<1, dim, VectorizedArray<double>> temp_surf_ten;

                    for (unsigned int i = 0; i < dim; ++i)
                      for (unsigned int j = 0; j < dim; ++j)
                        temp_surf_ten[i] = (i == j) ?
                                             -(make_vectorized_array<double>(1.) - n[i] * n[j]) *
                                               d_alpha0 * grad_T[j] :
                                             (n[i] * n[j]) * d_alpha0 * grad_T[j];

                    const auto alpha = compare_and_apply_mask<SIMDComparison::less_than>(
                      T,
                      T0,
                      VectorizedArray<double>(alpha0),
                      VectorizedArray<double>(alpha0) -
                        VectorizedArray<double>(d_alpha0) * (T - T0));

                    for (unsigned int v = 0; v < VectorizedArray<double>::size(); ++v)
                      Assert(alpha[v] >= 0.0,
                             ExcMessage(
                               "The surface tension coefficient tends to be negative in "
                               "some regions. Check the value of the temperature dependent surface "
                               "tension coefficient."));

                    surface_tension.submit_value(alpha * n * curvature.get_value(q_index) +
                                                   temp_surf_ten,
                                                 q_index);
                  }
                surface_tension.integrate_scatter(true, false, force_rhs);
              }
          },
          force_rhs,
          level_set_as_heaviside,
          zero_out);

        solution_curvature.zero_out_ghosts();
      }

      /**
       * The temperature (member variable of this class) is calculated using an analytic expression
       * for a given heaviside representation of a level set field. The resulting DoF vector will be
       * based on the same DoFHandler as the level set field.
       */
      void
      compute_temperature_vector(const VectorType &level_set_as_heaviside)
      {
        level_set_as_heaviside.update_ghost_values();

        scratch_data->initialize_dof_vector(temperature, temp_dof_idx);
        scratch_data->initialize_dof_vector(solid, temp_dof_idx);

        FEValues<dim> fe_values(scratch_data->get_mapping(),
                                scratch_data->get_dof_handler(temp_dof_idx).get_fe(),
                                scratch_data->get_quadrature(temp_quad_idx),
                                update_values | update_gradients | update_quadrature_points |
                                  update_JxW_values);

        const unsigned int dofs_per_cell = scratch_data->get_n_dofs_per_cell(temp_dof_idx);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        std::map<types::global_dof_index, Point<dim>> support_points;
        DoFTools::map_dofs_to_support_points(scratch_data->get_mapping(),
                                             scratch_data->get_dof_handler(temp_dof_idx),
                                             support_points);

        for (const auto &cell : scratch_data->get_dof_handler(temp_dof_idx).active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(local_dof_indices);
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  temperature[local_dof_indices[i]] =
                    analytical_temperature_field(support_points[local_dof_indices[i]],
                                                 level_set_as_heaviside[local_dof_indices[i]]);
                  solid[local_dof_indices[i]] =
                    is_solid_region(support_points[local_dof_indices[i]]);
                }
            }

        temperature.compress(VectorOperation::insert);
        solid.compress(VectorOperation::insert);

        scratch_data->get_constraint(temp_dof_idx).distribute(temperature);
        scratch_data->get_constraint(temp_dof_idx).distribute(solid);

        level_set_as_heaviside.zero_out_ghosts();
      }

      void
      set_flow_field_in_solid_regions_to_zero(const DoFHandler<dim> &    flow_dof_handler,
                                              AffineConstraints<double> &flow_constraints)
      {
        const unsigned int dofs_per_cell = flow_dof_handler.get_fe().n_dofs_per_cell();


        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        std::map<types::global_dof_index, Point<dim>> support_points;
        DoFTools::map_dofs_to_support_points(scratch_data->get_mapping(),
                                             flow_dof_handler,
                                             support_points);

        IndexSet flow_locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(flow_dof_handler, flow_locally_relevant_dofs);

        AffineConstraints<double> solid_constraints;
        solid_constraints.reinit(flow_locally_relevant_dofs);
        DoFTools::make_hanging_node_constraints(flow_dof_handler, solid_constraints);

        for (const auto &cell : flow_dof_handler.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(local_dof_indices);

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                if (is_solid_region(support_points[local_dof_indices[i]]))
                  {
                    solid_constraints.add_line(local_dof_indices[i]);
                  }
            }

        solid_constraints.close();
        flow_constraints.merge(solid_constraints,
                               AffineConstraints<double>::MergeConflictBehavior::left_object_wins);
      }

      void
      set_level_set(VectorType &level_set)
      {
        scratch_data->initialize_dof_vector(liquid, temp_dof_idx);
        level_set.update_ghost_values();

        FEValues<dim> fe_values(scratch_data->get_mapping(),
                                scratch_data->get_dof_handler(temp_dof_idx).get_fe(),
                                scratch_data->get_quadrature(temp_quad_idx),
                                update_values | update_gradients | update_quadrature_points |
                                  update_JxW_values);

        const unsigned int dofs_per_cell = scratch_data->get_n_dofs_per_cell(temp_dof_idx);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        std::map<types::global_dof_index, Point<dim>> support_points;
        DoFTools::map_dofs_to_support_points(scratch_data->get_mapping(),
                                             scratch_data->get_dof_handler(temp_dof_idx),
                                             support_points);

        for (const auto &cell : scratch_data->get_dof_handler(temp_dof_idx).active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(local_dof_indices);
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  liquid[local_dof_indices[i]] =
                    is_liquid_region(support_points[local_dof_indices[i]]);
                  if (liquid[local_dof_indices[i]])
                    level_set[local_dof_indices[i]] = 1.0;
                  else if (is_solid_region(support_points[local_dof_indices[i]]))
                    level_set[local_dof_indices[i]] = 0.0;
                  else
                    level_set[local_dof_indices[i]] = -1.0;
                }
            }
        level_set.zero_out_ghosts();
      }

      void
      remove_the_level_set_from_solid_regions(const DoFHandler<dim> &    level_set_dof_handler,
                                              AffineConstraints<double> &level_set_constraints)


      {
        FEValues<dim> fe_values(scratch_data->get_mapping(),
                                level_set_dof_handler.get_fe(),
                                scratch_data->get_quadrature(flow_quad_idx),
                                update_values);

        const unsigned int dofs_per_cell = level_set_dof_handler.get_fe().n_dofs_per_cell();

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        std::map<types::global_dof_index, Point<dim>> support_points;
        DoFTools::map_dofs_to_support_points(scratch_data->get_mapping(),
                                             level_set_dof_handler,
                                             support_points);

        AffineConstraints<double> solid_constraints;

        IndexSet ls_locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(level_set_dof_handler, ls_locally_relevant_dofs);

        solid_constraints.reinit(ls_locally_relevant_dofs);
        for (const auto &cell : level_set_dof_handler.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(local_dof_indices);

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                if (is_solid_region(support_points[local_dof_indices[i]]))
                  solid_constraints.add_line(local_dof_indices[i]);
            }
        level_set_constraints.merge(
          solid_constraints, AffineConstraints<double>::MergeConflictBehavior::right_object_wins);
        level_set_constraints.close();
      }

      void
      reinit()
      {
        scratch_data->initialize_dof_vector(temperature, temp_dof_idx);
        scratch_data->initialize_dof_vector(solid, temp_dof_idx);
      }

      void
      attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
      {
        temperature.update_ghost_values();
        solid.update_ghost_values();
        vectors.push_back(&temperature);
        vectors.push_back(&solid);
      }

    private:
      void
      set_melt_pool_parameters(const Parameters<double> &data_in)
      {
        mp_data   = data_in.mp;
        flow_data = data_in.flow;
      }

      double
      analytical_temperature_field(Point<dim> point, const double phi)
      {
        if (mp_data.temperature_formulation == "analytical")
          {
            // The temperature function below is derived from the publication on
            // "Heat Source Modeling in Selective Laser Melting" by E. Mirkoohi, D. E. Seivers,
            //  H. Garmestani and S. Y. Liang
            //
            //  In order to capture anisotropic temperature fields, a modification is introduced.
            const double indicator = UtilityFunctions::CharacteristicFunctions::heaviside(phi, 0.5);


            double laser_intensity = 1.0;
            if (mp_data.laser_power_over_time == "ramp")
              {
                AssertThrow(
                  mp_data.laser_power_end_time > mp_data.laser_power_start_time,
                  ExcMessage(
                    "For the temporal ramp distribution of the laser power,"
                    " the parameter laser power end time must be larger than laser power start time."));
                laser_intensity = (time - mp_data.laser_power_start_time) /
                                  (mp_data.laser_power_end_time - mp_data.laser_power_start_time);
                laser_intensity = std::min(std::max(0.0, laser_intensity), 1.0);
              }


            const double &P  = mp_data.laser_power * laser_intensity;
            const double &v  = mp_data.scan_speed;
            const double &T0 = mp_data.ambient_temperature;
            const double &absorptivity =
              (indicator == 1) ? mp_data.liquid.absorptivity : mp_data.gas.absorptivity;
            const double &conductivity =
              (indicator == 1) ? mp_data.liquid.conductivity : mp_data.gas.conductivity;
            const double &capacity =
              (indicator == 1) ? mp_data.liquid.capacity : mp_data.gas.capacity;
            const double density = flow_data.density + flow_data.density_difference * indicator;

            const double thermal_diffusivity = conductivity / (density * capacity);

            // modify temperature profile to be anisotropic
            for (int d = 0; d < dim - 1; d++)
              point[d] *= mp_data.temperature_x_to_y_ratio;

            double R = point.distance(laser_center);

            if (R == 0.0)
              R = 1e-16;
            double T = P * absorptivity / (4 * numbers::PI * R * conductivity) *
                         std::exp(-v * (R) / (2. * thermal_diffusivity)) +
                       T0;
            return (T > mp_data.max_temperature) ? mp_data.max_temperature : T;
          }
        else
          AssertThrow(false, ExcNotImplemented());
      }

      /**
       *  This function determines for a given point, whether it belongs to the solid domain.
       *
       *  WARNING: All points above (component dim-1) the center point of the laser source are
       *  automatically identified as gaseous parts. Thus, this function has to be modified when the
       *  initial interface between the feedstock and the ambient gas is not planar.
       */
      bool
      is_solid_region(const Point<dim> point)
      {
        if (mp_data.liquid.melt_pool_radius == 0)
          return false;
        else if (point[dim - 1] >= laser_center[dim - 1])
          return false;
        else
          {
            Point<dim> shifted_center =
              MeltPoolDG::UtilityFunctions::convert_string_coords_to_point<dim>(
                mp_data.melt_pool_center);

            if (mp_data.melt_pool_shape == "parabola")
              {
                if (dim == 2)
                  {
                    const double sign =
                      -2 * mp_data.liquid.melt_pool_radius * (point[1] - shifted_center[1]) +
                      std::pow(point[0] - shifted_center[0], 2);
                    return (sign >= 0) ? true : false;
                  }
                else
                  AssertThrow(false, ExcMessage("not implemented"));
              }
            else if (mp_data.melt_pool_shape == "ellipse")
              return UtilityFunctions::DistanceFunctions::ellipsoidal_manifold<dim>(
                       point,
                       shifted_center,
                       mp_data.liquid.melt_pool_radius,
                       mp_data.liquid.melt_pool_depth) > 0 ?
                       false :
                       true;
            else if (mp_data.melt_pool_shape == "temperature_dependent")
              return (analytical_temperature_field(point, 1.0 /* is_liquid */) >=
                      mp_data.liquid.melting_point) ?
                       false :
                       true;
            else
              AssertThrow(false, ExcMessage("not implemented"));
          }
      }

      /*
       * check if a given point is liquid
       */
      bool
      is_liquid_region(const Point<dim> point)
      {
        if (mp_data.liquid.melt_pool_radius == 0)
          return false;
        else if (point[dim - 1] >= laser_center[dim - 1])
          return false;
        else
          {
            Point<dim> shifted_center =
              MeltPoolDG::UtilityFunctions::convert_string_coords_to_point<dim>(
                mp_data.melt_pool_center);

            if (mp_data.melt_pool_shape == "parabola")
              {
                if (dim == 2)
                  {
                    const double sign =
                      -2 * mp_data.liquid.melt_pool_radius * (point[1] - shifted_center[1]) +
                      std::pow(point[0] - shifted_center[0], 2);
                    return (sign >= 0) ? false : true;
                  }
                else
                  AssertThrow(false, ExcMessage("not implemented"));
              }
            else if (mp_data.melt_pool_shape == "ellipse")
              return UtilityFunctions::DistanceFunctions::ellipsoidal_manifold<dim>(
                       point,
                       shifted_center,
                       mp_data.liquid.melt_pool_radius,
                       mp_data.liquid.melt_pool_depth) > 0 ?
                       true :
                       false;
            else if (mp_data.melt_pool_shape == "temperature_dependent")
              return (analytical_temperature_field(point, 1.0 /* is_liquid */) >=
                      mp_data.liquid.melting_point) ?
                       true :
                       false;
            else
              AssertThrow(false, ExcMessage("not implemented"));
          }
      }
      /**
       *  This function computes the recoil pressure coefficient for a given temperature value
       *  dependent on the input parameters.
       */
      inline double
      compute_recoil_pressure_coefficient(const double T)
      {
        return mp_data.recoil_pressure_constant *
               std::exp(-mp_data.recoil_pressure_temperature_constant *
                        (1. / T - 1. / mp_data.boiling_temperature));
      }

    private:
      std::shared_ptr<const ScratchData<dim>> scratch_data;
      /*
       *  Center of the laser
       */
      Point<dim> laser_center;

      /*
       *  Based on the following indices the correct DoFHandler or quadrature rule from
       *  ScratchData<dim> object is selected. This is important when ScratchData<dim> holds
       *  multiple DoFHandlers, quadrature rules, etc.
       */
      unsigned int ls_dof_idx;
      unsigned int flow_vel_dof_idx;
      unsigned int flow_quad_idx;
      unsigned int temp_dof_idx;
      unsigned int temp_quad_idx;

      /*
       *  current time
       */
      double time;
    };
  } // namespace MeltPool
} // namespace MeltPoolDG
