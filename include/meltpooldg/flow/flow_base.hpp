/* ---------------------------------------------------------------------
 *
 * Author: Peter Münch, Magdalena Schreter, TUM, October 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/dofs/dof_handler.h>

namespace MeltPoolDG
{
  namespace Flow
  {
    using namespace dealii;

    template <int dim>
    class FlowBase
    {
    public:
      virtual void
      solve() = 0;

      virtual void
      get_velocity(LinearAlgebra::distributed::Vector<double> &vec) const = 0;

      virtual const DoFHandler<dim> &
      get_dof_handler_velocity() const = 0;

      virtual const unsigned int &
      get_dof_handler_idx_velocity() const = 0;

      virtual const AffineConstraints<double> &
      get_constraints_velocity() const = 0;

      virtual AffineConstraints<double> &
      get_constraints_velocity() = 0;

      virtual const LinearAlgebra::distributed::Vector<double> &
      get_pressure() const = 0;

      virtual const DoFHandler<dim> &
      get_dof_handler_pressure() const = 0;

      virtual const AffineConstraints<double> &
      get_constraints_pressure() const = 0;

      virtual AffineConstraints<double> &
      get_constraints_pressure() = 0;

      virtual void
      set_force_rhs(const LinearAlgebra::distributed::BlockVector<double> &vec) = 0;

      virtual VectorizedArray<double> &
      get_density(const unsigned int cell, const unsigned int q) = 0;

      virtual const VectorizedArray<double> &
      get_density(const unsigned int cell, const unsigned int q) const = 0;

      virtual VectorizedArray<double> &
      get_viscosity(const unsigned int cell, const unsigned int q) = 0;

      virtual const VectorizedArray<double> &
      get_viscosity(const unsigned int cell, const unsigned int q) const = 0;

      virtual void
      attach_vectors_u(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) = 0;

      virtual void
      attach_vectors_p(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) = 0;
    };

  } // namespace Flow
} // namespace MeltPoolDG
