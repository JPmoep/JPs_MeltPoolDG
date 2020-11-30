#pragma once

// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <iostream>
// MeltPoolDG
#include <meltpooldg/interface/simulationbase.hpp>

namespace MeltPoolDG
{
  namespace Simulation
  {
    namespace RecoilPressure
    {
      using namespace dealii;

      template <int dim>
      class InitialValuesLS : public Function<dim>
      {
      public:
        InitialValuesLS()
          : Function<dim>()
        {}

        double
        value(const Point<dim> &p, const unsigned int /*component*/) const
        {
          const double x_half_domain_size = 200e-6;
          const double y_min = -100e-6;
          const double y_max = 40e-6;
          Point<dim>   lower_left =
            dim == 2 ? Point<dim>(-x_half_domain_size, y_min) :
                       Point<dim>(-x_half_domain_size, x_half_domain_size, y_min);
          Point<dim> upper_right = dim == 2 ? Point<dim>(x_half_domain_size, y_max) :
                                              Point<dim>(x_half_domain_size, x_half_domain_size, y_max);

          return UtilityFunctions::CharacteristicFunctions::sgn(
            UtilityFunctions::DistanceFunctions::rectangular_manifold<dim>(p,
                                                                           lower_left,
                                                                           upper_right));
        }
      };

      /*
       *      This class collects all relevant input data for the level set simulation
       */

      template <int dim>
      class SimulationRecoilPressure : public SimulationBase<dim>
      {
      public:
        SimulationRecoilPressure(std::string parameter_file, const MPI_Comm mpi_communicator)
          : SimulationBase<dim>(parameter_file, mpi_communicator)
        {
          this->set_parameters();
        }

        void
        create_spatial_discretization() override
        {
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

          const double x_half_domain_size = 200e-6;
          const double y_half_domain_size = 200e-6;
          
          if constexpr (dim == 2)
            {
              // create mesh
              const Point<dim> bottom_left = Point<dim>(-x_half_domain_size, -y_half_domain_size/2);
              const Point<dim> top_right   = Point<dim>(x_half_domain_size, y_half_domain_size);

              GridGenerator::hyper_rectangle(*this->triangulation, bottom_left, top_right);
              this->triangulation->refine_global(this->parameters.base.global_refinements);
            }
          else if constexpr (dim == 3)
            {
              // create mesh
              const double z_half_domain_size = 200e-6;

              const Point<dim> bottom_left =
                Point<dim>(-x_half_domain_size, -y_half_domain_size/2, -z_half_domain_size);
              const Point<dim> top_right =
                Point<dim>(x_half_domain_size, y_half_domain_size, z_half_domain_size);

              GridGenerator::hyper_rectangle(*this->triangulation, bottom_left, top_right);
              this->triangulation->refine_global(this->parameters.base.global_refinements);
            }
          else
            {
              AssertThrow(false, ExcNotImplemented());
            }
        }

        void
        set_boundary_conditions() override
        {
          this->attach_no_slip_boundary_condition(0, "navier_stokes_u");
          this->attach_fix_pressure_constant_condition(0, "navier_stokes_p");

          // auto dirichlet = std::make_shared<Functions::ConstantFunction<dim>>(-1.0);
          // this->attach_dirichlet_boundary_condition(0, dirichlet, "level_set");
          // this->attach_dirichlet_boundary_condition(2, dirichlet, "level_set");
        }

        void
        set_field_conditions() override
        {
          this->attach_initial_condition(std::make_shared<InitialValuesLS<dim>>(), "level_set");
          this->attach_initial_condition(std::shared_ptr<Function<dim>>(
                                           new Functions::ZeroFunction<dim>(dim)),
                                         "navier_stokes_u");
        }
      };

    } // namespace RecoilPressure
  }   // namespace Simulation
} // namespace MeltPoolDG
