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
    namespace SpuriousCurrents
    {
      using namespace dealii;

      template <int dim>
      class InitializePhi : public Function<dim>
      {
      public:
        InitializePhi(const double eps)
          : Function<dim>()
          , eps(eps)
        {}
        virtual double
        value(const Point<dim> &p, const unsigned int component = 0) const
        {
          (void)component;

          // set radius of bubble to 0.5, slightly shifted away from the center
          Point<dim> center;
          for (unsigned int d = 0; d < dim; ++d)
            center[d] = 0.02 + 0.01 * d;

          return UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function(
            UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, center, 0.5), eps);

          // Alternative if no signed distance function should be used in the beginning:
          //
          // return UtilityFunctions::CharacteristicFunctions::sgn(
          // UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, center, 0.5));
        }

        double eps = 0.0;
      };

      /*
       *      This class collects all relevant input data for the level set simulation
       */

      template <int dim>
      class SimulationSpuriousCurrents : public SimulationBase<dim>
      {
      public:
        SimulationSpuriousCurrents(std::string parameter_file, const MPI_Comm mpi_communicator)
          : SimulationBase<dim>(parameter_file, mpi_communicator)
        {
          this->set_parameters();
        }


        void
        create_spatial_discretization() override
        {
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

          if constexpr (dim == 2)
            {
              GridGenerator::hyper_cube(*this->triangulation, -2.5, 2.5);
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
          this->attach_dirichlet_boundary_condition(
            0, std::make_shared<Functions::ConstantFunction<dim>>(-1), "level_set");
          this->attach_no_slip_boundary_condition(0, "navier_stokes_u");
          this->attach_fix_pressure_constant_condition(0, "navier_stokes_p");
        }

        void
        set_field_conditions() override
        {
          double eps = 0.0;
          if (this->parameters.reinit.implementation == "adaflo" ||
              this->parameters.ls.implementation == "adaflo")
            eps = this->parameters.reinit.constant_epsilon > 0.0 ?
                    this->parameters.reinit.constant_epsilon :
                    GridTools::minimal_cell_diameter(*this->triangulation) /
                      this->parameters.base.degree / std::sqrt(dim);
          else
            eps = this->parameters.reinit.constant_epsilon > 0.0 ?
                    this->parameters.reinit.constant_epsilon :
                    GridTools::minimal_cell_diameter(*this->triangulation) / std::sqrt(dim) *
                      this->parameters.reinit.scale_factor_epsilon;

          AssertThrow(eps > 0, ExcNotImplemented());

          this->attach_initial_condition(std::make_shared<InitializePhi<dim>>(eps), "level_set");
          this->attach_initial_condition(std::make_shared<Functions::ZeroFunction<dim>>(dim),
                                         "navier_stokes_u");
        }
      };

    } // namespace SpuriousCurrents
  }   // namespace Simulation
} // namespace MeltPoolDG
