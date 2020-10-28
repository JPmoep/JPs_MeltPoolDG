#pragma once

#ifdef MELT_POOL_DG_WITH_ADAFLO

#  include <adaflo/navier_stokes.h>
#  include <adaflo/parameters.h>

#  include <meltpooldg/flow/adaflo_wrapper_parameters.hpp>
# include <meltpooldg/interface/scratch_data.hpp>

namespace MeltPoolDG
{
namespace Flow
{
  template <int dim>
  class InflowVelocity : public Function<dim>
  {
  public:
    InflowVelocity (const double time,
                    const bool fluctuating)
      :
      Function<dim>(dim, time),
      fluctuating(fluctuating)
    {}

    virtual void vector_value(const Point<dim> &p,
                                    Vector<double>   &values) const
  {
    AssertDimension (values.size(), dim);

    // inflow velocity according to Schaefer & Turek
    const double Um = (dim == 2 ? 1.5 : 2.25);
    const double H = 0.41;
    double coefficient = Utilities::fixed_power<dim-1>(4.) * Um / Utilities::fixed_power<2*dim-2>(H);
    values(0) = coefficient * p[1] * (H-p[1]);
    if (dim == 3)
      values(0) *= p[2] * (H-p[2]);
    if (fluctuating)
      values(0) *= std::sin(this->get_time()*numbers::PI/8.);
    for (unsigned int d=1; d<dim; ++d)
      values(d) = 0;
  }

  private:
    const bool fluctuating;
  };

    template <int dim>
    class AdafloWrapper
    {
    public:

      /**
       * Constructor.
       */
      template<int space_dim, typename number, typename VectorizedArrayType>
      AdafloWrapper(ScratchData<dim, space_dim, number, VectorizedArrayType> & scratch_data, 
                    const AdafloWrapperParameters & parameters_in) : navier_stokes(
                    parameters_in.get_parameters(),
                    *const_cast<parallel::distributed::Triangulation<dim> *>(dynamic_cast<const parallel::distributed::Triangulation<dim> *>(&scratch_data.get_triangulation()))
                    )
      {
        // set boundary conditions
        navier_stokes.set_no_slip_boundary(0);
        navier_stokes.set_velocity_dirichlet_boundary(1, std::shared_ptr<Function<dim> >(new InflowVelocity<dim>(0., true)));

        navier_stokes.set_open_boundary_with_normal_flux(2, std::shared_ptr<Function<dim> > (new Functions::ZeroFunction<dim>(1)));

        // set initial condition
        navier_stokes.setup_problem(InflowVelocity<dim>(0., false));
      }

      /**
       * Solver time step
       */
      void
      solve()
      {          
        navier_stokes.advance_time_step();
      }

      const LinearAlgebra::distributed::Vector<double> &
      get_velocity() const
      {
        return navier_stokes.solution.block(0);
      }

      void
      set_surface_tension(const LinearAlgebra::distributed::Vector<double> & vec)
      {
        navier_stokes.user_rhs.block(0).copy_locally_owned_data_from(vec);
      }

    private:
      /**
       * Reference to the actual Navier-Stokes solver from adaflo
       */
      NavierStokes<dim> navier_stokes;
    };

    /**
     * Dummy specialization for 1. Needed to be able to compile
     * since the adaflo Navier-Stokes solver is not compiled for
     * 1D - due to the dependcy to parallel::distributed::Triangulation
     * and p4est.
     */
    template <>
    class AdafloWrapper<1>
    {
    public:
      /**
       * Dummy constructor.
       */
      template<int space_dim, typename number, typename VectorizedArrayType>
      AdafloWrapper(ScratchData<1, space_dim, number, VectorizedArrayType> & scratch_data, 
                    const AdafloWrapperParameters parameters_in)
      {
        (void) scratch_data;
        (void) parameters_in;

        AssertThrow(false, ExcNotImplemented ());
      }


      const LinearAlgebra::distributed::Vector<double> &
      get_velocity() const
      {
        AssertThrow(false, ExcNotImplemented ());
        return dummy;
      }

      void
      set_surface_tension(const LinearAlgebra::distributed::Vector<double> & )
      {
        AssertThrow(false, ExcNotImplemented ());
      }

      void
      set_density_viscosity()
      {
        
      }

      void
      solve()
      {
        AssertThrow(false, ExcNotImplemented ());
      }

      private:
        LinearAlgebra::distributed::Vector<double> dummy;
    };

} // namespace Flow
} // namespace MeltPoolDG

#endif
