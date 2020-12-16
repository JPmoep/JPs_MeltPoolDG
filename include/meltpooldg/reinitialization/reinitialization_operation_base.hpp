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
  namespace Reinitialization
  {
    using namespace dealii;

    template <int dim>
    class ReinitializationOperationBase
    {
    public:
      virtual void
      solve(const double dt) = 0;

      virtual void
      initialize(const std::shared_ptr<const ScratchData<dim>> &   scratch_data_in,
                 const LinearAlgebra::distributed::Vector<double> &solution_level_set_in,
                 const Parameters<double> &                        data_in,
                 const unsigned int                                re_dof_idx_in,
                 const unsigned int                                re_quad_idx_in)
      {
        (void)scratch_data_in;
        (void)solution_level_set_in;
        (void)data_in;
        (void)re_dof_idx_in;
        (void)re_quad_idx_in;
        AssertThrow(false, ExcNotImplemented());
      }

      virtual void
      reinit() = 0;

      virtual void
      update_initial_solution(
        const LinearAlgebra::distributed::Vector<double> &solution_level_set_in) = 0;

      virtual const LinearAlgebra::distributed::Vector<double> &
      get_level_set() const = 0;

      virtual LinearAlgebra::distributed::Vector<double> &
      get_level_set() = 0;

      virtual const LinearAlgebra::distributed::BlockVector<double> &
      get_normal_vector() const = 0;

      virtual void
      attach_vectors(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) = 0;
    };

  } // namespace Reinitialization
} // namespace MeltPoolDG
