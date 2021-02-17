/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, January 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/generic_linear_algebra.h>

#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Evaporation
{
  using namespace dealii;
  /**
   *     This module computes for a given evaporative mass flux the corresponding interface
   *     velocity according to
   *
   *     \f[ \boldsymbol{n}\cfrac{\dot{m}}{\rho} \f]
   *
   *     with the normal vector \f$\boldsymbol{n}\f$, the evaporative mass flux \f$\dot{m}\f$
   *     and the density \f$\rho\f$. One has to take care from which time step the normal
   *     vector is computed.
   */
  template <int dim>
  class EvaporationOperation
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

  public:
    EvaporationOperation(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
                         const VectorType &                             level_set_as_heaviside_in,
                         const BlockVectorType &                        normal_vector_in,
                         std::shared_ptr<SimulationBase<dim>>           base_in,
                         const unsigned int                             normal_dof_idx_in,
                         const unsigned int vel_hanging_nodes_dof_idx_in,
                         const unsigned int ls_hanging_nodes_dof_idx_in,
                         const unsigned int ls_quad_idx_in)
      : scratch_data(scratch_data_in)
      , evaporation_data(base_in->parameters.evapor)
      , level_set_as_heaviside(level_set_as_heaviside_in)
      , normal_vector(normal_vector_in)
      , normal_dof_idx(normal_dof_idx_in)
      , vel_hanging_nodes_dof_idx(vel_hanging_nodes_dof_idx_in)
      , ls_hanging_nodes_dof_idx(ls_hanging_nodes_dof_idx_in)
      , ls_quad_idx(ls_quad_idx_in)
    {
      reinit();
    }

    void
    reinit()
    {
      scratch_data->initialize_dof_vector(interface_velocity, vel_hanging_nodes_dof_idx);
    }

    void
    solve(const VectorType &fluid_velocity)
    {
      level_set_as_heaviside.update_ghost_values();
      normal_vector.update_ghost_values();
      fluid_velocity.update_ghost_values();

      FECellIntegrator<dim, 1, double> ls(scratch_data->get_matrix_free(),
                                          ls_hanging_nodes_dof_idx,
                                          ls_quad_idx);

      FECellIntegrator<dim, dim, double> vel(scratch_data->get_matrix_free(),
                                             vel_hanging_nodes_dof_idx,
                                             ls_quad_idx);

      FECellIntegrator<dim, dim, double> normal_vec(scratch_data->get_matrix_free(),
                                                    normal_dof_idx,
                                                    ls_quad_idx);

      evaporation_velocities.resize(scratch_data->get_matrix_free().n_cell_batches() *
                                    ls.n_q_points);


      for (unsigned int cell = 0; cell < scratch_data->get_matrix_free().n_cell_batches(); ++cell)
        {
          Tensor<1, dim, VectorizedArray<double>> *interface_vel = begin_interface_velocity(cell);

          ls.reinit(cell);
          ls.read_dof_values(level_set_as_heaviside);
          ls.evaluate(true, false);

          normal_vec.reinit(cell);
          normal_vec.read_dof_values(normal_vector);
          normal_vec.evaluate(true, false);

          vel.reinit(cell);
          vel.read_dof_values(fluid_velocity);
          vel.evaluate(true, false);

          for (unsigned int q_index = 0; q_index < ls.n_q_points; ++q_index)
            {
              // get a vector with 1 = gas phase and 0 = liquid phase
              auto is_gas_phase =
                (evaporation_data.ls_value_gas == 1.0) ?
                  UtilityFunctions::heaviside(ls.get_value(q_index), 0.0) :
                  std::abs(1. - UtilityFunctions::heaviside(ls.get_value(q_index), 0.0));

              // determine the density
              auto density =
                evaporation_data.density_liquid +
                (evaporation_data.density_gas - evaporation_data.density_liquid) * is_gas_phase;

              const auto n_phi =
                MeltPoolDG::VectorTools::normalize<dim>(normal_vec.get_value(q_index));
              interface_vel[q_index] = n_phi * evaporation_data.evaporative_mass_flux / density;

              // The normal vector field is oriented such that the normal vector points from
              // the negative level set value (= default for representing the gas phase) to the
              // positive value (= default for representing the liquid phase). Thus, in case the gas
              // phase corresponds to a level set value of 1, the sign of the normal vector has to
              // be changed.
              if (evaporation_data.ls_value_gas == 1.0)
                interface_vel[q_index] *= -1.0;

              interface_vel[q_index] +=
                MeltPoolDG::VectorTools::convert_to_vector<dim>(vel.get_value(q_index));
            }
        }

      level_set_as_heaviside.zero_out_ghosts();
      normal_vector.zero_out_ghosts();
      fluid_velocity.zero_out_ghosts();

      reinit();

      /**
       * write interface velocity to dof vector
       */
      UtilityFunctions::fill_dof_vector_from_cell_operation_vec<dim, dim>(
        interface_velocity,
        scratch_data->get_matrix_free(),
        vel_hanging_nodes_dof_idx,
        ls_quad_idx,
        scratch_data->get_fe(vel_hanging_nodes_dof_idx)
          .tensor_degree(), // fe_degree of the resulting vector
        scratch_data->get_fe(ls_hanging_nodes_dof_idx).tensor_degree() +
          1, // n_q_points_1d of cell operation
        [&](const unsigned int cell,
            const unsigned int quad) -> const Tensor<1, dim, VectorizedArray<double>> & {
          return begin_interface_velocity(cell)[quad];
        });

      scratch_data->get_constraint(vel_hanging_nodes_dof_idx).distribute(interface_velocity);
      interface_velocity.zero_out_ghosts();
    }

    void
    compute_mass_balance_source_term(VectorType &       mass_balance_rhs,
                                     const unsigned int pressure_dof_idx,
                                     const unsigned int pressure_quad_idx,
                                     bool               zero_out)
    {
      normal_vector.update_ghost_values();

      scratch_data->get_matrix_free().template cell_loop<VectorType, VectorType>(
        [&](const auto &matrix_free,
            auto &      force_rhs,
            const auto &level_set_as_heaviside,
            auto        macro_cells) {
          FECellIntegrator<dim, 1, double> heaviside(matrix_free,
                                                     ls_hanging_nodes_dof_idx,
                                                     pressure_quad_idx);

          FECellIntegrator<dim, dim, double> normal_vec(matrix_free,
                                                        normal_dof_idx,
                                                        pressure_quad_idx);
          FECellIntegrator<dim, 1, double>   mass_flux(matrix_free,
                                                     pressure_dof_idx,
                                                     pressure_quad_idx);

          for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
            {
              heaviside.reinit(cell);
              heaviside.read_dof_values_plain(level_set_as_heaviside);
              heaviside.evaluate(false, true);

              normal_vec.reinit(cell);
              normal_vec.read_dof_values_plain(normal_vector);
              normal_vec.evaluate(true, false);

              mass_flux.reinit(cell);

              for (unsigned int q_index = 0; q_index < mass_flux.n_q_points; ++q_index)
                {
                  const auto n_phi =
                    MeltPoolDG::VectorTools::normalize<dim>(normal_vec.get_value(q_index));

                  // the factor of 0.5 is needed to ensure that the integral of phi * n * 0.5
                  // over the volume is equal to 1 and represents the approximation of a delta
                  // function. @todo -- better solution?
                  mass_flux.submit_value((1. / evaporation_data.density_liquid -
                                          1. / evaporation_data.density_gas) *
                                           evaporation_data.evaporative_mass_flux *
                                           -heaviside.get_gradient(q_index) * n_phi * 0.5,
                                         q_index);
                }
              mass_flux.integrate_scatter(true, false, force_rhs);
            }
        },
        mass_balance_rhs,
        level_set_as_heaviside,
        zero_out);

      normal_vector.zero_out_ghosts();
    }

    inline Tensor<1, dim, VectorizedArray<double>> *
    begin_interface_velocity(const unsigned int macro_cell)
    {
      AssertIndexRange(macro_cell, scratch_data->get_matrix_free().n_cell_batches());
      AssertDimension(evaporation_velocities.size(),
                      scratch_data->get_matrix_free().n_cell_batches() *
                        scratch_data->get_matrix_free().get_n_q_points(ls_quad_idx));
      return &evaporation_velocities[scratch_data->get_matrix_free().get_n_q_points(ls_quad_idx) *
                                     macro_cell];
    }

    inline const Tensor<1, dim, VectorizedArray<double>> &
    begin_interface_velocity(const unsigned int macro_cell) const
    {
      AssertIndexRange(macro_cell, scratch_data->get_matrix_free().n_cell_batches());
      AssertDimension(evaporation_velocities.size(),
                      scratch_data->get_matrix_free().n_cell_batches() *
                        scratch_data->get_matrix_free().get_n_q_points(ls_quad_idx));
      return evaporation_velocities[scratch_data->get_matrix_free().get_n_q_points(ls_quad_idx) *
                                    macro_cell];
    }

    const LinearAlgebra::distributed::Vector<double> &
    get_interface_velocity() const
    {
      return interface_velocity;
    }

    LinearAlgebra::distributed::Vector<double> &
    get_interface_velocity()
    {
      return interface_velocity;
    }

    virtual void
    attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
    {
      interface_velocity.update_ghost_values();
      vectors.push_back(&interface_velocity);
    }

    void
    attach_output_vectors(DataOut<dim> &data_out) const
    {
      /*
       *  evaporation velocity
       */
      interface_velocity.update_ghost_values();

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        vector_component_interpretation(dim,
                                        DataComponentInterpretation::component_is_part_of_vector);

      data_out.add_data_vector(scratch_data->get_dof_handler(vel_hanging_nodes_dof_idx),
                               interface_velocity,
                               std::vector<std::string>(dim, "interface_velocity"),
                               vector_component_interpretation);
    }

  private:
    std::shared_ptr<const ScratchData<dim>> scratch_data;
    /**
     *  parameters controlling the evaporation
     */
    EvaporationData<double> evaporation_data;
    /**
     * references to solutions needed for the computation
     */
    const VectorType &     level_set_as_heaviside;
    const BlockVectorType &normal_vector;
    /**
     * select the relevant DoFHandlers and quadrature rules
     */
    const unsigned int normal_dof_idx;
    const unsigned int vel_hanging_nodes_dof_idx;
    const unsigned int ls_hanging_nodes_dof_idx;
    const unsigned int ls_quad_idx;
    /**
     * interface velocity at quadrature points
     */
    AlignedVector<Tensor<1, dim, VectorizedArray<double>>> evaporation_velocities;
    /**
     * interface velocity due to evaporation and flow
     */
    VectorType interface_velocity;
  };
} // namespace MeltPoolDG::Evaporation
