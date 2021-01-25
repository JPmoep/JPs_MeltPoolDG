/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, January 2021
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/generic_linear_algebra.h>

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
   *     vector is computed
   */
  template <int dim>
  class EvaporationOperation
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

  public:
    EvaporationOperation(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
                         const VectorType &                             velocity_in,
                         const VectorType &                             level_set_in,
                         const BlockVectorType &                        normal_vector_in,
                         std::shared_ptr<SimulationBase<dim>>           base_in,
                         const unsigned int                             normal_dof_idx_in,
                         const unsigned int vel_hanging_nodes_dof_idx_in,
                         const unsigned int ls_dof_idx_in,
                         const unsigned int ls_quad_idx_in)
      : scratch_data(scratch_data_in)
      , evaporation_data(base_in->parameters.evapor)
      , level_set(level_set_in)
      , normal_vector(normal_vector_in)
      , normal_dof_idx(normal_dof_idx_in)
      , vel_hanging_nodes_dof_idx(vel_hanging_nodes_dof_idx_in)
      , ls_dof_idx(ls_dof_idx_in)
      , ls_quad_idx(ls_quad_idx_in)
    {
      reinit();
    }

    void
    reinit()
    {
      scratch_data->initialize_dof_vector(evaporation_velocity, vel_hanging_nodes_dof_idx);
    }

    void
    solve()
    {
      level_set.update_ghost_values();
      normal_vector.update_ghost_values();
      reinit();
      evaporation_velocities.resize(scratch_data->get_matrix_free().n_cell_batches() * dim *
                                    scratch_data->get_matrix_free().get_n_q_points(ls_quad_idx));

      FECellIntegrator<dim, 1, double> ls(scratch_data->get_matrix_free(), ls_dof_idx, ls_quad_idx);

      FECellIntegrator<dim, dim, double> normal_vec(scratch_data->get_matrix_free(),
                                                    normal_dof_idx,
                                                    ls_quad_idx);

      for (unsigned int cell = 0; cell < scratch_data->get_matrix_free().n_cell_batches(); ++cell)
        {
          Tensor<1, dim, VectorizedArray<double>> *evapor_velocity = begin_interface_velocity(cell);

          ls.reinit(cell);
          ls.read_dof_values_plain(level_set);
          ls.evaluate(true, false);

          normal_vec.reinit(cell);
          normal_vec.read_dof_values_plain(normal_vector);
          normal_vec.evaluate(true, false);

          for (unsigned int q_index = 0; q_index < ls.n_q_points; ++q_index)
            {
              //@todo: comment
              auto is_one_phase =
                (evaporation_data.ls_value_gas == 1.0) ?
                  UtilityFunctions::heaviside(ls.get_value(q_index), 0.0) :
                  std::abs(1. - UtilityFunctions::heaviside(ls.get_value(q_index), 0.0));

              auto density =
                evaporation_data.density_liquid +
                (evaporation_data.density_gas - evaporation_data.density_liquid) * is_one_phase;

              evapor_velocity[q_index] =
                normal_vec.get_value(q_index) * evaporation_data.evaporative_mass_flux / density;

              if (evaporation_data.ls_value_gas == 1.0)
                evapor_velocity[q_index] *= -1.0;
            }
        }

      level_set.zero_out_ghosts();
      normal_vector.zero_out_ghosts();

      UtilityFunctions::fill_dof_vector_from_cell_operation_vec<dim, dim>(
        evaporation_velocity,
        scratch_data->get_matrix_free(),
        vel_hanging_nodes_dof_idx,
        ls_quad_idx,
        scratch_data->get_fe(vel_hanging_nodes_dof_idx).tensor_degree(),     // fe_degree
        scratch_data->get_fe(vel_hanging_nodes_dof_idx).tensor_degree() + 1, // n_q_points_1d
        [&](const unsigned int cell,
            const unsigned int quad) -> const Tensor<1, dim, VectorizedArray<double>> & {
          return begin_interface_velocity(cell)[quad];
        });

      scratch_data->get_constraint(vel_hanging_nodes_dof_idx).distribute(evaporation_velocity);
    }

    inline Tensor<1, dim, VectorizedArray<double>> *
    begin_interface_velocity(const unsigned int macro_cell)
    {
      AssertIndexRange(macro_cell, scratch_data->get_matrix_free().n_cell_batches());
      AssertDimension(evaporation_velocities.size(),
                      scratch_data->get_matrix_free().n_cell_batches() *
                        scratch_data->get_matrix_free().get_n_q_points(ls_quad_idx) * dim);
      return &evaporation_velocities[scratch_data->get_matrix_free().get_n_q_points(ls_quad_idx) *
                                     dim * macro_cell];
    }

    inline const Tensor<1, dim, VectorizedArray<double>> &
    begin_interface_velocity(const unsigned int macro_cell) const
    {
      AssertIndexRange(macro_cell, scratch_data->get_matrix_free().n_cell_batches());
      AssertDimension(evaporation_velocities.size(),
                      scratch_data->get_matrix_free().n_cell_batches() *
                        scratch_data->get_matrix_free().get_n_q_points(ls_quad_idx) * dim);
      return evaporation_velocities[scratch_data->get_matrix_free().get_n_q_points(ls_quad_idx) *
                                    dim * macro_cell];
    }

  public:
    const LinearAlgebra::distributed::Vector<double> &
    get_evaporation_velocity() const
    {
      return evaporation_velocity;
    }

    LinearAlgebra::distributed::Vector<double> &
    get_evaporation_velocity()
    {
      return evaporation_velocity;
    }

    virtual void
    attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
    {
      evaporation_velocity.update_ghost_values();
      vectors.push_back(&evaporation_velocity);
    }

    void
    attach_output_vectors(DataOut<dim> &data_out) const
    {
      /*
       *  evaporation velocity
       */
      MeltPoolDG::VectorTools::update_ghost_values(evaporation_velocity);

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        vector_component_interpretation(dim,
                                        DataComponentInterpretation::component_is_part_of_vector);

      data_out.add_data_vector(scratch_data->get_dof_handler(vel_hanging_nodes_dof_idx),
                               evaporation_velocity,
                               std::vector<std::string>(dim, "evaporation_velocity"),
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
    const VectorType &     level_set;
    const BlockVectorType &normal_vector;
    /**
     * select the relevant DoFHandlers and quadrature rules
     */
    unsigned int normal_dof_idx;
    unsigned int vel_hanging_nodes_dof_idx;
    unsigned int ls_dof_idx;
    unsigned int ls_quad_idx;
    /**
     * interface velocity at quadrature points
     */
    AlignedVector<Tensor<1, dim, VectorizedArray<double>>> evaporation_velocities;
    /**
     * velocity due to evaporation
     */
    VectorType evaporation_velocity;
  };
} // namespace MeltPoolDG::Evaporation
