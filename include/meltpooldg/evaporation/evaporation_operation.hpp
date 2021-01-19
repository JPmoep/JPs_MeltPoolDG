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

namespace MeltPoolDG
{
  namespace Evaporation
  {
    using namespace dealii;
    /*
     *     This module computes for a given evaporation rate m_dot the interface
     *     velocity.
     */
    template <int dim>
    class EvaporationOperation
    {
    private:
      using VectorType      = LinearAlgebra::distributed::Vector<double>;
      using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    public:
      Evaporation(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
                  const VectorType &                             velocity_in,
                  const BlockVectorType &                        normal_vector_in,
                  const VectorType &                             density_liquid_in,
                  std::shared_ptr<SimulationBase<dim>>           base_in,
                  const unsigned int                             normal_dof_idx_in,
                  const unsigned int                             vel_dof_idx_in,
                  const unsigned int                             vel_quad_idx_in,
                  const unsigned int                             density_dof_idx_in)
        : scratch_data(scratch_data_in)
        , evaporation_data(base_in->parameters.evaporation_data)
        , advection_velocity(velocity_in)
        , normal_vector(normal_vector_in)
        , density_liquid(density_liquid_in)
        , normal_dof_idx(normal_dof_idx_in)
        , vel_dof_idx(vel_dof_idx_in)
        , vel_quad_idx(vel_quad_idx_in)
        , density_dof_idx(density_dof_idx_in)

      {}

      void
      reinit()
      {}

      void
      solve()
      {
        scratch_data->get_matrix_free().template cell_loop<VectorType, VectorType>(
          [&](const auto &matrix_free, auto &dst, const auto &src, auto macro_cells) {
            FECellIntegrator<dim, 1, double> velocity(matrix_free, vel_dof_idx, vel_quad_idx);
            FECellIntegrator<dim, 1, double> interface_velocity(matrix_free,
                                                                vel_dof_idx,
                                                                vel_quad_idx);

            FECellIntegrator<dim, dim, double> normal_vec(matrix_free,
                                                          normal_dof_idx,
                                                          vel_quad_idx);

            FECellIntegrator<dim, dim, double> density_liquid(matrix_free,
                                                              density_dof_idx,
                                                              vel_quad_idx);

            for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
              {
                velocity.reinit(cell);
                velocity.read_dof_values_plain(src);
                velocity.evaluate(true, false);

                interface_velocity.reinit(cell);

                normal_vec.reinit(cell);
                normal_vec.read_dof_values_plain(normal_vector);
                normal_vec.evaluate(true, false);

                density_liquid.reinit(cell);
                density_liquid.read_dof_values_plain(density_liquid);
                density_liquid.evaluate(true, false);

                for (unsigned int q_index = 0; q_index < interface_velocity.n_q_points; ++q_index)
                  {
                    interface_velocity.submit_value(velocity.get_value(q_index) +
                                                      normal_vec.get_value(q_index) *
                                                        evaporation_data.evaporative_mass_flux /
                                                        density_liquid.get_value(q_index),
                                                    q_index);
                  }
                interface_velocity.integrate_scatter(true, false, dst);
              }
          },
          interface_velocity,
          advection_velocity,
          true);
      }


    public:
      const LinearAlgebra::distributed::Vector<double> &
      get_interface_velocity() const
      {
        return interface_velocity;
      }

      virtual void
      attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
      {}

    private:
      std::shared_ptr<const ScratchData<dim>> scratch_data;
      /*
       *  All the necessary parameters are stored in this vector.
       */
      EvaporationData<double> evaporation_data;
      /*
       * this vector refers to the advection velocity
       */
      const VectorType &     advection_velocity;
      const BlockVectorType &normal_vector;
      const VectorType &     density_liquid;
      /*
       * select the relevant DoFHandlers and quadrature rules
       */
      unsigned int normal_dof_idx;
      unsigned int vel_dof_idx;
      unsigned int vel_quad_idx;
      unsigned int density_dof_idx;
      /*
       * this vector holds the interface velocity
       */
      VectorType interface_velocity;
    };
  } // namespace Evaporation
} // namespace MeltPoolDG
