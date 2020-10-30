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
      set_surface_tension(const LinearAlgebra::distributed::BlockVector<double> &vec) = 0;
      
      virtual void
      update_phases(const LinearAlgebra::distributed::Vector<double> & vec) = 0;
    };

  } // namespace Flow
} // namespace MeltPoolDG
