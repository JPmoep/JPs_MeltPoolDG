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
      using VectorType       = LinearAlgebra::distributed::Vector<double>;
      using BlockVectorType  = LinearAlgebra::distributed::BlockVector<double>;

    public:
      /*
       *    This is the primary solution variable of this module, which will be also publically
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
                 const unsigned int                             quad_idx_in
                  )
      {
        scratch_data  = scratch_data_in;
        dof_idx       = dof_no_bc_idx_in;
        quad_idx      = quad_idx_in;
        /*
         *  set the advection diffusion data
         */
        mp_data = data_in.mp;
        /*
         *  set the parameters for the melt pool operation
         */
        set_melt_pool_parameters(data_in);
        /*
         * 
         */

        scratch_data->initialize_dof_vector(temperature, dof_idx);
        dealii::VectorTools::project(scratch_data->get_mapping(),
                                     scratch_data->get_dof_handler(dof_idx),
                                     scratch_data->get_constraint(dof_idx),
                                     scratch_data->get_quadrature(),
                                     Functions::ConstantFunction<dim>(mp_data.ambient_temperature),//*base_in->get_initial_condition("level_set"),
                                     temperature);
      }


      void
      compute_recoil_pressure_force(BlockVectorType &force_rhs,
                                    const VectorType& level_set_as_heaviside,
                                    bool zero_out = true)
      {
        level_set_as_heaviside.update_ghost_values();
        scratch_data->get_matrix_free().template cell_loop<BlockVectorType, std::nullptr_t>(
          [&](const auto &matrix_free, auto &force_rhs, const auto &, auto macro_cells) {
            FECellIntegrator<dim, 1, double> level_set(matrix_free, dof_idx, quad_idx);

            FECellIntegrator<dim, dim, double> recoil_pressure(matrix_free,
                                                               dof_idx,
                                                               quad_idx);

            for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
              {
                level_set.reinit(cell);
                level_set.gather_evaluate(level_set_as_heaviside, true, true);

                recoil_pressure.reinit(cell);

                for (unsigned int q_index = 0; q_index < recoil_pressure.n_q_points; ++q_index)
                  {
                    const auto phi = level_set.get_value(q_index);
                    
                    VectorizedArray<double> recoil_pressure_coefficient;  
          
                    for (unsigned int v=0; v<VectorizedArray<double>::size(); ++v)
                    {
                      Point<dim> p;
                      for (unsigned int d=0; d<dim; ++d)
                        p[d] = recoil_pressure.quadrature_point(q_index)[d][v];

                      recoil_pressure_coefficient[v] = compute_recoil_pressure_coefficient(compute_temperature(p, phi[v]));
                    }
                    recoil_pressure.submit_value(
                        recoil_pressure_coefficient
                        *
                        level_set.get_gradient(q_index)
                        ,
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
      //void
      //solve(const double time, const VectorType &solution_level_set)
      //{
        //solution_level_set.update_ghost_values();

        //if (mp_data.do_print_l2norm)
          //{
            //const ConditionalOStream &pcout = scratch_data->get_pcout();
            //pcout << "| Temperature |T|2 = " << std::setw(15) << std::left << std::setprecision(10)
                  //<< temperature.l2_norm() << std::endl;
          //}
        
        //compute_temperature(solution_level_set);

        //solution_level_set.zero_out_ghosts();
      //}

    private:
      void
      set_melt_pool_parameters(const Parameters<double> &data_in)
      {
        mp_data = data_in.mp;
        flow_data = data_in.flow;
      }


      double
      compute_temperature(const Point<dim> point, const double phi)
      {
          if (mp_data.temperature_formulation == "analytical")
          {

            const double indicator = UtilityFunctions::CharacteristicFunctions::heaviside(phi, 0.0);
            // this is the temperature function according to Heat Source Modeling in Selective Laser Melting, 
            // E. Mirkoohi, D. E. Seivers, H. Garmestani and S. Y. Liang
            const double& P            = mp_data.laser_power; // @todo: make dependent from input parameters
            const double& v            = mp_data.scan_speed;
            const double& T0           = mp_data.ambient_temperature;
            const double& absorptivity = (indicator == 1) ? mp_data.liquid.absorptivity : mp_data.gas.absorptivity;
            const double& conductivity = (indicator == 1) ? mp_data.liquid.conductivity : mp_data.gas.conductivity;
            const double& capacity     = (indicator == 1) ? mp_data.liquid.capacity     : mp_data.gas.capacity;
            const double density       = flow_data.density + flow_data.density_difference * indicator;

            const double thermal_diffusivity = conductivity / (density * capacity);
            constexpr double pi = std::acos(-1); // @todo move to utility function
            const double R = 0;

            return P * absorptivity / (4 * pi * R ) * std::exp( - v * (R-point[dim-1]) / (2.*thermal_diffusivity)) + T0;
          }
          else
            AssertThrow(false, ExcNotImplemented());
      }

      double
      compute_recoil_pressure_coefficient(const double T)
      {
        return mp_data.recoil_pressure_constant * std::exp(-mp_data.recoil_pressure_temperature_constant*(1./T-1./mp_data.boiling_temperature));
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
