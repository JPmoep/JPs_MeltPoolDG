/* ---------------------------------------------------------------------
 *
 * Author: Peter Münch, Magdalena Schreter, TUM, October 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

namespace MeltPoolDG
{
  namespace Flow
  {
    class FlowBase
    {
    public:
      virtual void
      solve() = 0;

      virtual void
      get_velocity(LinearAlgebra::distributed::BlockVector<double> &vec) const = 0;
      
      virtual void
      get_pressure(LinearAlgebra::distributed::Vector<double> &vec) const = 0;

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
    };

  } // namespace Flow
} // namespace MeltPoolDG
