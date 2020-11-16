#pragma once
// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/simplex/grid_generator.h>
// c++
#include <cmath>
#include <iostream>
// MeltPoolDG
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/utilities/utilityfunctions.hpp>

namespace MeltPoolDG
{
  namespace Simulation
  {
    namespace ReinitCircle
    {
      using namespace dealii;
      /*
       * this function specifies the initial field of the level set equation
       */

      template <int dim>
      class InitializePhi : public Function<dim>
      {
      public:
        InitializePhi()
          : Function<dim>()
        {}
        virtual double
        value(const Point<dim> &p, const unsigned int component = 0) const
        {
          (void)component;
          Point<dim>   center = dim == 1 ? Point<dim>(0.0) : Point<dim>(0.0, 0.5);
          const double radius = 0.25;
          return UtilityFunctions::CharacteristicFunctions::sgn(
            UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, center, radius));
        }
      };

      template <int dim>
      class ExactSolution : public Function<dim>
      {
      public:
        ExactSolution(const double eps)
          : Function<dim>()
          , eps_interface(eps)
        {}

        double
        value(const Point<dim> &p, const unsigned int component = 0) const
        {
          (void)component;
          Point<dim>   center = Point<dim>(0.0, 0.5);
          const double radius = 0.25;
          return UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function(
            UtilityFunctions::DistanceFunctions::spherical_manifold(p, center, radius),
            eps_interface);
        }

      private:
        double eps_interface;
      };
      /*
       *      This class collects all relevant input data for the level set simulation
       */

      template <int dim>
      class SimulationReinit : public SimulationBase<dim>
      {
      public:
        SimulationReinit(std::string parameter_file, const MPI_Comm mpi_communicator)
          : SimulationBase<dim>(parameter_file, mpi_communicator)
        {
          this->set_parameters();
        }

        void
        create_spatial_discretization()
        {
          if (dim == 1 || this->parameters.base.do_simplex)
            {
              AssertDimension(Utilities::MPI::n_mpi_processes(this->mpi_communicator), 1);
              this->triangulation = std::make_shared<Triangulation<dim>>();
            }
          else
            {
              this->triangulation =
                std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
            }

#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
          if (this->parameters.base.do_simplex)
            {
              GridGenerator::subdivided_hyper_cube_with_simplices(
                *this->triangulation,
                Utilities::pow(2, this->parameters.base.global_refinements),
                left_domain,
                right_domain);
            }
          else
#endif
            {
              GridGenerator::hyper_cube(*this->triangulation, left_domain, right_domain);
              this->triangulation->refine_global(this->parameters.base.global_refinements);
            }
        }

        void
        set_boundary_conditions()
        {}

        void
        set_field_conditions()
        {
          this->attach_initial_condition(std::make_shared<InitializePhi<dim>>(), "level_set");
          this->attach_exact_solution(std::make_shared<ExactSolution<dim>>(0.01), "level_set");
        }

      private:
        double left_domain  = -1.0;
        double right_domain = 1.0;
      };

    } // namespace ReinitCircle
  }   // namespace Simulation
} // namespace MeltPoolDG
