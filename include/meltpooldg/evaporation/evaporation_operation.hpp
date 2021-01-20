/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, January 2021
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
      EvaporationOperation(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
                           const VectorType &                             velocity_in,
                           const VectorType &                             level_set_in,
                           const BlockVectorType &                        normal_vector_in,
                           std::shared_ptr<SimulationBase<dim>>           base_in,
                           const unsigned int                             normal_dof_idx_in,
                           const unsigned int                             vel_dof_idx_in,
                           const unsigned int                             ls_dof_idx_in,
                           const unsigned int                             ls_quad_idx_in)
        : scratch_data(scratch_data_in)
        , evaporation_data(base_in->parameters.evapor)
        , advection_velocity(velocity_in)
        , level_set(level_set_in)
        , normal_vector(normal_vector_in)
        , normal_dof_idx(normal_dof_idx_in)
        , vel_dof_idx(vel_dof_idx_in)
        , ls_dof_idx(ls_dof_idx_in)
        , ls_quad_idx(ls_quad_idx_in)
      {
        reinit();
      }

      void
      reinit()
      {
        scratch_data->initialize_dof_vector(evaporation_velocity, vel_dof_idx);
      }

      void
      solve()
      {
        level_set.update_ghost_values();
        evaporation_velocity.update_ghost_values();

        evaporation_velocities.resize(scratch_data->get_matrix_free().n_cell_batches() * dim *
                                      scratch_data->get_matrix_free().get_n_q_points(ls_quad_idx));

        FECellIntegrator<dim, 1, double> ls(scratch_data->get_matrix_free(),
                                            ls_dof_idx,
                                            ls_quad_idx);

        FECellIntegrator<dim, dim, double> normal_vec(scratch_data->get_matrix_free(),
                                                      normal_dof_idx,
                                                      ls_quad_idx);

        for (unsigned int cell = 0; cell < scratch_data->get_matrix_free().n_cell_batches(); ++cell)
          {
            Tensor<1, dim, VectorizedArray<double>> *evapor_velocity =
              begin_interface_velocity(cell);

            ls.reinit(cell);
            ls.read_dof_values_plain(level_set);
            ls.evaluate(true, false);

            normal_vec.reinit(cell);
            normal_vec.read_dof_values_plain(normal_vector);
            normal_vec.evaluate(true, false);

            for (unsigned int q_index = 0; q_index < ls.n_q_points; ++q_index)
              {
                auto is_liquid = UtilityFunctions::heaviside(ls.get_value(q_index), 0.0);
                auto density =
                  evaporation_data.density_liquid +
                  (evaporation_data.density_liquid - evaporation_data.density_gas) * is_liquid;

                evapor_velocity[q_index] =
                  normal_vec.get_value(q_index) *
                  make_vectorized_array<double>(evaporation_data.evaporative_mass_flux) / density;
              }
          }

        level_set.zero_out_ghosts();

        // UtilityFunctions::fill_dof_vector_from_cell_operation<dim, dim>(
        // evaporation_velocity,
        // scratch_data->get_matrix_free(),
        // vel_dof_idx,
        // ls_quad_idx,
        // scratch_data->get_fe(vel_dof_idx).tensor_degree(), // fe_degree
        // scratch_data->get_fe(vel_dof_idx).tensor_degree() + 1, // n_q_points_1d
        // dim, // n_components
        //[&](const unsigned int cell, const unsigned int quad) -> const
        // Tensor<1,dim,VectorizedArray<double>> & { return begin_interface_velocity(cell)[quad];
        //});
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

      virtual void
      attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors)
      {
        (void)vectors;
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

        data_out.add_data_vector(scratch_data->get_dof_handler(vel_dof_idx),
                                 evaporation_velocity,
                                 std::vector<std::string>(dim, "evaporation_velocity"),
                                 vector_component_interpretation);
      }

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
      const VectorType &     level_set;
      const BlockVectorType &normal_vector;
      /*
       * select the relevant DoFHandlers and quadrature rules
       */
      unsigned int normal_dof_idx;
      unsigned int vel_dof_idx;
      unsigned int ls_dof_idx;
      unsigned int ls_quad_idx;
      /*
       * this vector holds the interface velocity
       */
      AlignedVector<Tensor<1, dim, VectorizedArray<double>>> evaporation_velocities;
      /*
       * this vector holds the velocity term due to evaporation
       */
      VectorType evaporation_velocity;
    };
  } // namespace Evaporation
} // namespace MeltPoolDG
