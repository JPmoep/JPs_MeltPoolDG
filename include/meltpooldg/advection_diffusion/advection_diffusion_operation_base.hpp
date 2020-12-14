/* ---------------------------------------------------------------------
 *
 * Author: Peter Münch, Magdalena Schreter, TUM, October 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/dofs/dof_handler.h>

#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/interface/parameters.hpp>

namespace MeltPoolDG
{
  namespace AdvectionDiffusion
  {
    using namespace dealii;

    template <int dim>
    class AdvectionDiffusionOperationBase
    {
    public:
      virtual void
      solve(const double dt, const LinearAlgebra::distributed::BlockVector<double>& velocity) = 0;
    
      virtual void
      initialize(const std::shared_ptr<const ScratchData<dim>> &scratch_data_in,
                 const LinearAlgebra::distributed::Vector<double>& solution_advected_field_in,
                 const Parameters<double> &                     data_in,
                 const unsigned int                             dof_idx_in,
                 const unsigned int                             dof_no_bc_idx_in,
                 const unsigned int                             quad_idx_in,
                 const unsigned int                             velocity_dof_idx_in)
      {
        (void)scratch_data_in;
        (void)solution_advected_field_in;
        (void)data_in;
        (void)dof_idx_in;
        (void)dof_no_bc_idx_in;
        (void)quad_idx_in;
        (void)velocity_dof_idx_in;
        AssertThrow(false, ExcNotImplemented());
      }
    };

  } // namespace Flow
} // namespace MeltPoolDG
