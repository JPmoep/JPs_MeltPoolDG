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
      AdafloWrapper(const Triangulation<dim> & tria)
      {

        FlowParameters params;

        dynamic_cast<parallel::distributed::Triangulation<dim> &>(&tria);

        navier_stokes = std::make_shared<avierStokes<dim>>(params, dynamic_cast<>);

        if constexpr (dim > 1)
          {
            

            parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

            
          }
      }

    private:
      shared_ptr<NavierStokes<dim>> navier_stokes;
    };

    template <>
    class AdafloWrapper<1>
    {

    public:
      AdafloWrapper()
      {
        AssertThrow(false, ExcNotImplemented ());
      }

      private:
    }

  } // namespace Flow
} // namespace MeltPoolDG

#endif
