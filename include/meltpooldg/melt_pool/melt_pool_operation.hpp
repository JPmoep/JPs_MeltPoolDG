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
      VectorType recoil_pressure;
      VectorType temperature;
      /*
       *  All the necessary parameters are stored in this struct.
       */
      MeltPoolData<double> mp_data;
      FlowData<double>     flow_data;

      MeltPoolOperation() = default;

      void
      initialize(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
                 const Parameters<double> &                     data_in,
                 const unsigned int                             dof_no_bc_idx_in,
                 const unsigned int                             quad_idx_in)
      {
        scratch_data = scratch_data_in;
        dof_idx      = dof_no_bc_idx_in;
        quad_idx     = quad_idx_in;
        /*
         *  set the advection diffusion data
         */
        mp_data = data_in.mp;
        /*
         *  set the parameters for the melt pool operation
         */
        set_melt_pool_parameters(data_in);
        /*
         *  Initialize the temperature field
         */
        scratch_data->initialize_dof_vector(temperature, dof_idx);
        dealii::VectorTools::project(scratch_data->get_mapping(),
                                     scratch_data->get_dof_handler(dof_idx),
                                     scratch_data->get_constraint(dof_idx),
                                     scratch_data->get_quadrature(),
                                     Functions::ConstantFunction<dim>(mp_data.ambient_temperature),
                                     temperature);
      }

      /**
       * The force contribution of the recoil pressure due to evaporation is computed. The model of
       * S.I. Anisimov and V.A. Khokhlov (1995) is considered. The consideration of any other model
       * is however possible. First, the temperature is updated and second, the recoil pressure is
       * computed.
       */

      void
      compute_recoil_pressure_force(BlockVectorType & force_rhs,
                                    const VectorType &level_set_as_heaviside,
                                    bool              zero_out = true)
      {
        level_set_as_heaviside.update_ghost_values();

        compute_temperature_vector(level_set_as_heaviside);

        scratch_data->get_matrix_free().template cell_loop<BlockVectorType, std::nullptr_t>(
          [&](const auto &matrix_free, auto &force_rhs, const auto &, auto macro_cells) {
            FECellIntegrator<dim, 1, double>   level_set(matrix_free, dof_idx, quad_idx);
            FECellIntegrator<dim, dim, double> recoil_pressure(matrix_free, dof_idx, quad_idx);

            FECellIntegrator<dim, 1, double> temperature_val(matrix_free, dof_idx, quad_idx);

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

                    VectorizedArray<double> recoil_pressure_coefficient;

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
          nullptr,
          zero_out);

        level_set_as_heaviside.zero_out_ghosts();
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

        scratch_data->initialize_dof_vector(temperature, dof_idx);

        FEValues<dim> fe_values(scratch_data->get_mapping(),
                                scratch_data->get_dof_handler(dof_idx).get_fe(),
                                scratch_data->get_quadrature(quad_idx),
                                update_values | update_gradients | update_quadrature_points |
                                  update_JxW_values);

        const unsigned int dofs_per_cell = scratch_data->get_n_dofs_per_cell(this->dof_idx);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        std::map<types::global_dof_index, Point<dim>> support_points;
        DoFTools::map_dofs_to_support_points(scratch_data->get_mapping(),
                                             scratch_data->get_dof_handler(dof_idx),
                                             support_points);

        for (const auto &cell :
             scratch_data->get_dof_handler(this->dof_idx).active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(local_dof_indices);
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                temperature[local_dof_indices[i]] =
                  analytical_temperature_field(support_points[local_dof_indices[i]],
                                               level_set_as_heaviside[local_dof_indices[i]]);
            }

        temperature.compress(VectorOperation::insert);

        level_set_as_heaviside.zero_out_ghosts();
      }

    private:
      void
      set_melt_pool_parameters(const Parameters<double> &data_in)
      {
        mp_data   = data_in.mp;
        flow_data = data_in.flow;
      }


      double
      analytical_temperature_field(const Point<dim> point, const double phi)
      {
        if (mp_data.temperature_formulation == "analytical")
          {
            // this is the temperature function according to Heat Source Modeling in Selective Laser
            // Melting, E. Mirkoohi, D. E. Seivers, H. Garmestani and S. Y. Liang
            Point<dim> laser_center;
            for (unsigned int d = 0; d < dim; ++d)
              laser_center[d] = 0.0;
            const double indicator = UtilityFunctions::CharacteristicFunctions::heaviside(phi, 0.0);
            const double &P  = mp_data.laser_power; // @todo: make dependent from input parameters
            const double &v  = mp_data.scan_speed;
            const double &T0 = mp_data.ambient_temperature;
            const double &absorptivity =
              (indicator == 1) ? mp_data.liquid.absorptivity : mp_data.gas.absorptivity;
            const double &conductivity =
              (indicator == 1) ? mp_data.liquid.conductivity : mp_data.gas.conductivity;
            const double &capacity =
              (indicator == 1) ? mp_data.liquid.capacity : mp_data.gas.capacity;
            const double density = flow_data.density + flow_data.density_difference * indicator;

            const double     thermal_diffusivity = conductivity / (density * capacity);
            constexpr double pi                  = std::acos(-1); // @todo move to utility function
            const double     R                   = point.distance(laser_center);

            if (R == 0.0)
              return T0;
            else
              return P * absorptivity / (4 * pi * R) *
                       std::exp(-v * (R - point[dim - 1]) / (2. * thermal_diffusivity)) +
                     T0;
          }
        else
          AssertThrow(false, ExcNotImplemented());
      }


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
       *  Based on the following indices the correct DoFHandler or quadrature rule from
       *  ScratchData<dim> object is selected. This is important when ScratchData<dim> holds
       *  multiple DoFHandlers, quadrature rules, etc.
       */
      unsigned int dof_idx;
      unsigned int quad_idx;
    };
  } // namespace MeltPool
} // namespace MeltPoolDG
