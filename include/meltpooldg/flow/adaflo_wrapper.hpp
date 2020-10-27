#pragma once

#ifdef MELT_POOL_DG_WITH_ADAFLO

#  include <adaflo/navier_stokes.h>
#  include <adaflo/parameters.h>

namespace MeltPoolDG
{
  namespace Flow
  {
    template <int dim>
    class AdafloWrapper
    {
    public:
      AdafloWrapper()
      {
        if constexpr (dim > 1)
          {
            FlowParameters params;

            parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

            NavierStokes<dim> navier_stokes(params, tria);
          }
      }

    private:
    };

  } // namespace Flow
} // namespace MeltPoolDG

#endif
