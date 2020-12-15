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
  namespace Curvature
  {
    using namespace dealii;

    template <int dim>
    class CurvatureOperationBase
    {
    public:
      virtual void
      solve(const LinearAlgebra::distributed::Vector<double> &advected_field) = 0;

      virtual void
      initialize(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
                 const Parameters<double> &                     data_in,
                 const unsigned int                             dof_idx_in,
                 const unsigned int                             quad_idx_in)
      {
        (void)scratch_data_in;
        (void)data_in;
        (void)dof_idx_in;
        (void)quad_idx_in;
        AssertThrow(false, ExcNotImplemented());
      }

      virtual const LinearAlgebra::distributed::Vector<double> &
      get_curvature() const = 0;
    };

  } // namespace NormalVector
} // namespace MeltPoolDG
