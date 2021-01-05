/* ---------------------------------------------------------------------
 *
 * Author: Peter Münch, Magdalena Schreter, TUM, October 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/dofs/dof_handler.h>

#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/interface/scratch_data.hpp>

namespace MeltPoolDG
{
  namespace AdvectionDiffusion
  {
    using namespace dealii;

    template <int dim>
    class AdvectionDiffusionOperationBase
    {
    public:
      AdvectionDiffusionData<double> advec_diff_data;

      virtual void
      solve(const double dt, const LinearAlgebra::distributed::Vector<double> &velocity) = 0;

      virtual void
      initialize(const std::shared_ptr<const ScratchData<dim>> &   scratch_data_in,
                 const LinearAlgebra::distributed::Vector<double> &solution_advected_field_in,
                 const Parameters<double> &                        data_in,
                 const unsigned int                                advec_diff_dof_idx_in,
                 const unsigned int                                advec_diff_hanging_nodes_idx_in,
                 const unsigned int                                advec_diff_quad_idx_in,
                 const unsigned int                                velocity_dof_idx_in)
      {
        (void)scratch_data_in;
        (void)solution_advected_field_in;
        (void)data_in;
        (void)advec_diff_dof_idx_in;
        (void)advec_diff_hanging_nodes_idx_in;
        (void)advec_diff_quad_idx_in;
        (void)velocity_dof_idx_in;
        AssertThrow(false, ExcNotImplemented());
      }

      virtual void
      reinit()
      {
        AssertThrow(false, ExcNotImplemented());
      }

      virtual void
      set_initial_condition(
        const LinearAlgebra::distributed::Vector<double> &initial_solution_advected_field,
        const LinearAlgebra::distributed::Vector<double> &velocity_vec_in)
      {
        (void)initial_solution_advected_field;
        (void)velocity_vec_in;
        AssertThrow(false, ExcNotImplemented());
      }

      virtual const LinearAlgebra::distributed::Vector<double> &
      get_advected_field() const = 0;

      virtual LinearAlgebra::distributed::Vector<double> &
      get_advected_field() = 0;

      virtual const LinearAlgebra::distributed::Vector<double> &
      get_advected_field_old() const = 0;

      virtual LinearAlgebra::distributed::Vector<double> &
      get_advected_field_old() = 0;

      virtual void
      attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) = 0;

      virtual void
      attach_output_vectors(DataOut<dim> &data_out) const = 0;
    };

  } // namespace AdvectionDiffusion
} // namespace MeltPoolDG
