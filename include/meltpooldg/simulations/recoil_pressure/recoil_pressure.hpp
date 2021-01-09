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
        InitialValuesLS(const double x_min,
                        const double x_max,
                        const double y_min,
                        const double y_interface)
          : Function<dim>()
          , x_min(x_min)
          , x_max(x_max)
          , y_min(y_min)
          , y_interface(y_interface)
        {}

        double
        value(const Point<dim> &p, const unsigned int /*component*/) const
        {
          Point<dim> lower_left =
            dim == 2 ? Point<dim>(x_min, y_min) : Point<dim>(x_min, x_min, y_min);
          Point<dim> upper_right =
            dim == 2 ? Point<dim>(x_max, y_interface) : Point<dim>(x_max, x_max, y_interface);

          return UtilityFunctions::CharacteristicFunctions::sgn(
            UtilityFunctions::DistanceFunctions::rectangular_manifold<dim>(p,
                                                                           lower_left,
                                                                           upper_right));
        }
        double x_min, x_max, y_min, y_interface;
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

          const double &x_min = this->parameters.mp.domain_x_min;
          const double &x_max = this->parameters.mp.domain_x_max;
          const double &y_min = this->parameters.mp.domain_y_min;
          const double &y_max = this->parameters.mp.domain_y_max;

          if (dim == 2)
            {
              // create mesh
              const Point<dim> bottom_left(x_min, y_min);
              const Point<dim> top_right(x_max, y_max);
#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
              if (this->parameters.base.do_simplex)
                {
                  unsigned int refinement =
                    Utilities::pow(2, this->parameters.base.global_refinements);
                  GridGenerator::subdivided_hyper_rectangle_with_simplices(*this->triangulation,
                                                                           {refinement, refinement},
                                                                           bottom_left,
                                                                           top_right);
                }
              else
#endif
                {
                  GridGenerator::hyper_rectangle(*this->triangulation, bottom_left, top_right);
                  this->triangulation->refine_global(this->parameters.base.global_refinements);
                }
            }
          else if (dim == 3)
            {
              // create mesh
              const Point<dim> bottom_left(x_min, x_min, y_min);
              const Point<dim> top_right(x_max, x_max, y_max);
#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
              if (this->parameters.base.do_simplex)
                {
                  unsigned int refinement =
                    Utilities::pow(2, this->parameters.base.global_refinements);
                  GridGenerator::subdivided_hyper_rectangle_with_simplices(*this->triangulation,
                                                                           {refinement, refinement},
                                                                           bottom_left,
                                                                           top_right);
                }
              else
#endif
                {
                  GridGenerator::hyper_rectangle(*this->triangulation, bottom_left, top_right);
                  this->triangulation->refine_global(this->parameters.base.global_refinements);
                }
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
        }

        void
        set_field_conditions() override
        {
          auto laser_center = MeltPoolDG::UtilityFunctions::convert_string_coords_to_point<dim>(
            this->parameters.mp.laser_center);
          this->attach_initial_condition(
            std::make_shared<InitialValuesLS<dim>>(this->parameters.mp.domain_x_min,
                                                   this->parameters.mp.domain_x_max,
                                                   this->parameters.mp.domain_y_min,
                                                   laser_center[dim - 1]),
            "level_set");
          this->attach_initial_condition(std::shared_ptr<Function<dim>>(
                                           new Functions::ZeroFunction<dim>(dim)),
                                         "navier_stokes_u");
        }
      };

    } // namespace RecoilPressure
  }   // namespace Simulation
} // namespace MeltPoolDG
