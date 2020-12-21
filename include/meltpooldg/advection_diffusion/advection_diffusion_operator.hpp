/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
// MeltPoolDG
#include <meltpooldg/advection_diffusion/advection_diffusion_operation.hpp>
#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/utilities/fe_integrator.hpp>

namespace MeltPoolDG
{
  namespace AdvectionDiffusion
  {
    static std::map<std::string, double> get_generalized_theta = {
      {"explicit_euler", 0.0},
      {"implicit_euler", 1.0},
      {"crank_nicolson", 0.5},
    };

    using namespace dealii;

    template <int dim, typename number = double>
    class AdvectionDiffusionOperator
      : public OperatorBase<number,
                            LinearAlgebra::distributed::Vector<number>,
                            LinearAlgebra::distributed::Vector<number>>
    {
    private:
      using VectorType          = LinearAlgebra::distributed::Vector<number>;
      using BlockVectorType     = LinearAlgebra::distributed::BlockVector<number>;
      using SparseMatrixType    = TrilinosWrappers::SparseMatrix;
      using VectorizedArrayType = VectorizedArray<number>;
      using vector              = Tensor<1, dim, VectorizedArray<number>>;
      using scalar              = VectorizedArray<number>;

    public:
      // clang-format off
    AdvectionDiffusionOperator( const ScratchData<dim>               &scratch_data_in, 
                                const VectorType                     &advection_velocity_in,
                                const AdvectionDiffusionData<number> &data_in,
                                const unsigned int                   dof_idx_in,
                                const unsigned int                   quad_idx_in, 
                                const unsigned int                   velocity_dof_idx_in)
    : scratch_data        ( scratch_data_in       )
    , advection_velocity  ( advection_velocity_in )
    , data                ( data_in               )
    , velocity_dof_idx    ( velocity_dof_idx_in   )
    {
      this->reset_indices(dof_idx_in, quad_idx_in);
      /*
       *  convert the user input to the generalized theta parameter
       */
      if (get_generalized_theta.find(data.time_integration_scheme) != get_generalized_theta.end()) 
        theta = get_generalized_theta[data.time_integration_scheme];
      else
        AssertThrow(false, ExcMessage("Advection diffusion operator: Requested time integration scheme not supported."))
    }
      // clang-format on

      /*
       *    this is the matrix-based implementation of the rhs and the matrix
       *    @todo: this could be improved by using the WorkStream functionality of dealii
       */

      void
      assemble_matrixbased(const VectorType &advected_field_old,
                           SparseMatrixType &matrix,
                           VectorType &      rhs) const override
      {
        AssertThrow(data.diffusivity >= 0.0,
                    ExcMessage("Advection diffusion operator: diffusivity is smaller than zero!"));

        //// @todo: the case where the size of the velocity dof vector is
        //// different than the level set field must be added
        // AssertThrow(velocity_dof_idx == this->dof_idx,
        // ExcMessage("not implemented; try matrix free operation"));

        advected_field_old.update_ghost_values();

        const FEValuesExtractors::Vector velocities(0);

        FEValues<dim> advec_diff_values(scratch_data.get_mapping(),
                                        scratch_data.get_dof_handler(this->dof_idx).get_fe(),
                                        scratch_data.get_quadrature(this->quad_idx),
                                        update_values | update_gradients |
                                          update_quadrature_points | update_JxW_values);

        FEValues<dim> vel_values(scratch_data.get_mapping(),
                                 scratch_data.get_dof_handler(velocity_dof_idx).get_fe(),
                                 scratch_data.get_quadrature(this->quad_idx),
                                 update_values);

        const unsigned int dofs_per_cell = scratch_data.get_n_dofs_per_cell();

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double>     cell_rhs(dofs_per_cell);

        const unsigned int n_q_points = advec_diff_values.get_quadrature().size();

        std::vector<double>         phi_at_q(n_q_points);
        std::vector<Tensor<1, dim>> grad_phi_at_q(n_q_points, Tensor<1, dim>());

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        rhs    = 0.0;
        matrix = 0.0;

        std::vector<Tensor<1, dim>> a(n_q_points, Tensor<1, dim>());

        typename DoFHandler<dim>::active_cell_iterator vel_cell =
          scratch_data.get_dof_handler(velocity_dof_idx).begin_active();

        for (const auto &cell : scratch_data.get_dof_handler(this->dof_idx).active_cell_iterators())
          {
            if (cell->is_locally_owned())
              {
                cell_matrix = 0;
                cell_rhs    = 0;

                advec_diff_values.reinit(cell);
                advec_diff_values.get_function_values(advected_field_old, phi_at_q);
                advec_diff_values.get_function_gradients(advected_field_old, grad_phi_at_q);

                vel_values.reinit(vel_cell);
                std::vector<Tensor<1, dim>> a(n_q_points, Tensor<1, dim>());
                vel_values[velocities].get_function_values(advection_velocity, a);

                for (const unsigned int q_index : advec_diff_values.quadrature_point_indices())
                  {
                    for (const unsigned int i : advec_diff_values.dof_indices())
                      {
                        for (const unsigned int j : advec_diff_values.dof_indices())
                          {
                            auto velocity_grad_phi_j =
                              a[q_index] * advec_diff_values.shape_grad(j, q_index);
                            // clang-format off
                  cell_matrix( i, j ) += 
                              ( advec_diff_values.shape_value( i, q_index) 
                                 * 
                                 advec_diff_values.shape_value( j, q_index) 
                                 +
                                 theta * this->d_tau * ( data.diffusivity * 
                                                       advec_diff_values.shape_grad( i, q_index) * 
                                                       advec_diff_values.shape_grad( j, q_index) 
                                                       +
                                                       advec_diff_values.shape_value( i, q_index)  *
                                                       velocity_grad_phi_j )
                              ) * advec_diff_values.JxW(q_index);
                            // clang-format on
                          }

                        // clang-format off
                cell_rhs( i ) +=
                  (  advec_diff_values.shape_value( i, q_index) * phi_at_q[q_index]
                      - 
                     ( 1. - theta ) * this->d_tau * 
                       (
                         data.diffusivity * advec_diff_values.shape_grad( i, q_index) * grad_phi_at_q[q_index]
                         +
                         advec_diff_values.shape_value(i, q_index) * a[q_index] * grad_phi_at_q[q_index] 
                       )
                    ) * advec_diff_values.JxW(q_index) ;
                        // clang-format on
                      }
                  } // end gauss

                // assembly
                cell->get_dof_indices(local_dof_indices);

                scratch_data.get_constraint(this->dof_idx)
                  .distribute_local_to_global(
                    cell_matrix, cell_rhs, local_dof_indices, matrix, rhs);
              }
            ++vel_cell;
          }

        matrix.compress(VectorOperation::add);
        rhs.compress(VectorOperation::add);
      }

      /*
       *    matrix-free implementation
       */
      void
      vmult(VectorType &dst, const VectorType &src) const override
      {
        AssertThrow(this->d_tau > 0.0,
                    ExcMessage("advection diffusion operator: d_tau must be set"));


        scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
          [&](const auto &matrix_free, auto &dst, const auto &src, auto cell_range) {
            FECellIntegrator<dim, 1, number>   advected_field(matrix_free,
                                                            this->dof_idx,
                                                            this->quad_idx);
            FECellIntegrator<dim, dim, number> velocity(matrix_free,
                                                        velocity_dof_idx,
                                                        this->quad_idx);

            for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
              {
                advected_field.reinit(cell);
                advected_field.gather_evaluate(src, true, true);

                velocity.reinit(cell);
                velocity.read_dof_values_plain(advection_velocity);
                velocity.evaluate(true, false);

                for (unsigned int q_index = 0; q_index < advected_field.n_q_points; ++q_index)
                  {
                    const scalar phi      = advected_field.get_value(q_index);
                    const vector grad_phi = advected_field.get_gradient(q_index);

                    const scalar velocity_grad_phi =
                      scalar_product(map_to_vector(velocity.get_value(q_index)), grad_phi);

                    advected_field.submit_value(phi + this->d_tau * theta * velocity_grad_phi,
                                                q_index);
                    advected_field.submit_gradient(this->d_tau * theta * data.diffusivity *
                                                     grad_phi,
                                                   q_index);
                  }
                advected_field.integrate_scatter(true, true, dst);
              }
          },
          dst,
          src,
          true);
      }

      void
      create_rhs(VectorType &dst, const VectorType &src) const override
      {
        /*
         * This function creates the rhs of the advection-diffusion problem. When inhomogeneous
         * dirichlet BC are prescribed, the rhs vector is modified including BC terms. Thus the src
         * vector will NOT be zeroed during the cell_loop.
         */
        AssertThrow(this->d_tau > 0.0,
                    ExcMessage("advection diffusion operator: d_tau must be set"));

        scratch_data.get_matrix_free().template cell_loop<VectorType, VectorType>(
          [&](const auto &matrix_free, auto &dst, const auto &src, auto macro_cells) {
            FECellIntegrator<dim, 1, number, VectorizedArrayType> advected_field(matrix_free,
                                                                                 this->dof_idx,
                                                                                 this->quad_idx);
            FECellIntegrator<dim, dim, number>                    velocity(matrix_free,
                                                        velocity_dof_idx,
                                                        this->quad_idx);

            for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
              {
                advected_field.reinit(cell);
                advected_field.gather_evaluate(src, true, true);

                velocity.reinit(cell);
                velocity.read_dof_values_plain(advection_velocity);
                velocity.evaluate(true, false);

                for (unsigned int q_index = 0; q_index < advected_field.n_q_points; ++q_index)
                  {
                    scalar       phi      = advected_field.get_value(q_index);
                    const vector grad_phi = advected_field.get_gradient(q_index);

                    const scalar velocity_grad_phi =
                      scalar_product(map_to_vector(velocity.get_value(q_index)), grad_phi);
                    advected_field.submit_value(phi -
                                                  this->d_tau * (1. - theta) * velocity_grad_phi,
                                                q_index);

                    advected_field.submit_gradient(-this->d_tau * (1. - theta) * data.diffusivity *
                                                     grad_phi,
                                                   q_index);
                  }

                advected_field.integrate_scatter(true, true, dst);
              }
          },
          dst,
          src,
          false); // rhs should not be zeroed out in order to consider inhomogeneous dirichlet BC
      }

    private:
      static vector
      map_to_vector(const scalar &in)
      {
        vector vec;

        for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
          vec[0][v] = in[v];

        return vec;
      }

      static vector
      map_to_vector(const vector &in)
      {
        return in;
      }


    private:
      const ScratchData<dim> &              scratch_data;
      const VectorType &                    advection_velocity;
      const AdvectionDiffusionData<number> &data;
      const unsigned int                    velocity_dof_idx;
      double                                theta;
    };
  } // namespace AdvectionDiffusion
} // namespace MeltPoolDG
