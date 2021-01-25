#pragma once
// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/simplex/grid_generator.h>
// c++
#include <cmath>
#include <iostream>
// MeltPoolDG
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/simulations/simulation_factory.hpp>
#include <meltpooldg/utilities/utilityfunctions.hpp>

namespace MeltPoolDG::Simulation::StefansProblem
{
  using namespace dealii;
  using namespace MeltPoolDG::Simulation;


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
      Point<dim> lower_left = dim == 2 ? Point<dim>(x_min, y_min) : Point<dim>(x_min, x_min, y_min);
      Point<dim> upper_right =
        dim == 2 ? Point<dim>(x_max, y_interface) : Point<dim>(x_max, x_max, y_interface);

      return UtilityFunctions::CharacteristicFunctions::sgn(
        UtilityFunctions::DistanceFunctions::rectangular_manifold<dim>(p, lower_left, upper_right));
    }
    double x_min, x_max, y_min, y_interface;
  };

  template <int dim>
  class AdvectionField : public Function<dim>
  {
  public:
    AdvectionField()
      : Function<dim>(dim)
    {}

    double
    value(const Point<dim> &p, const unsigned int component) const override
    {
      (void)p;
      Tensor<1, dim> value_;

      if constexpr (dim == 2)
        {
          // const double x = p[0];
          // const double y = p[1];

          value_[0] = 0.0;
          value_[1] = 0.0;
        }
      else
        AssertThrow(false, ExcMessage("Advection field for dim!=2 not implemented"));

      return value_[component];
    }

    void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      for (unsigned int c = 0; c < this->n_components; ++c)
        values(c) = value(p, c);
    }
  };

  /*
   *      This class collects all relevant input data for the level set simulation
   */

  template <int dim>
  class SimulationStefansProblem : public SimulationBase<dim>
  {
  public:
    SimulationStefansProblem(std::string parameter_file, const MPI_Comm mpi_communicator)
      : SimulationBase<dim>(parameter_file, mpi_communicator)
    {
      this->set_parameters();
    }

    void
    create_spatial_discretization() override
    {
      if (this->parameters.base.do_simplex)
        {
          this->triangulation = std::make_shared<parallel::shared::Triangulation<dim>>(
            this->mpi_communicator,
            (::Triangulation<dim>::none),
            false,
            parallel::shared::Triangulation<dim>::Settings::partition_metis);
        }
      else
        {
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
        }

      if constexpr ((dim == 2) || (dim == 3))
        {
          // create mesh
          const Point<dim> bottom_left =
            (dim == 2) ? Point<dim>(x_min, y_min) : Point<dim>(x_min, x_min, y_min);
          const Point<dim> top_right =
            (dim == 2) ? Point<dim>(x_max, y_max) : Point<dim>(x_max, x_max, y_max);

#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
          if (this->parameters.base.do_simplex)
            {
              // create mesh
              std::vector<unsigned int> subdivisions(
                dim, 5 * Utilities::pow(2, this->parameters.base.global_refinements));
              subdivisions[dim - 1] *= 2;

              GridGenerator::subdivided_hyper_rectangle_with_simplices(*this->triangulation,
                                                                       subdivisions,
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
    set_boundary_conditions() final
    {
      /*
       *  create a pair of (boundary_id, dirichlet_function)
       */

      constexpr types::boundary_id lower_bc = 1;
      constexpr types::boundary_id upper_bc = 2;

      if (this->parameters.evapor.ls_value_liquid == -1)
        {
          this->attach_dirichlet_boundary_condition(
            lower_bc, std::make_shared<Functions::ConstantFunction<dim>>(1.0), "level_set");
        }
      else if (this->parameters.evapor.ls_value_liquid == 1)
        {
          this->attach_dirichlet_boundary_condition(
            upper_bc, std::make_shared<Functions::ConstantFunction<dim>>(-1.0), "level_set");
        }


      /*
       *  mark inflow edges with boundary label (no boundary on outflow edges must be prescribed
       *  due to the hyperbolic nature of the analyzed problem)
       *
                      fix
       (0,1)  +---------------+ (1,1)
              |    ls=-1      |
              |               |
       sym    |               |  sym
              |               |
              |               |
              |    ls=1       |
              +---------------+
       * (0,1)      fix       (1,0)
       */
      if constexpr (dim == 2)
        {
          for (auto &face : this->triangulation->active_face_iterators())
            if ((face->at_boundary()))
              {
                if (face->center()[1] == y_min)
                  face->set_boundary_id(lower_bc);
                else if (face->center()[1] == y_max)
                  face->set_boundary_id(upper_bc);
              }
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
    }

    void
    set_field_conditions() final
    {
      this->attach_initial_condition(
        std::make_shared<InitialValuesLS<dim>>(x_min, x_max, y_min, y_interface), "level_set");
      this->attach_advection_field(std::make_shared<AdvectionField<dim>>(), "level_set");
    }

  private:
    const double x_min       = 0.0;
    const double x_max       = 1.0;
    const double y_min       = 0.0;
    const double y_max       = 1.0;
    const double y_interface = 0.5;
  };

  // const static bool isRegistered_1d = MeltPoolDG::SimulationFactory<1,1>::print();
  // const static bool isRegistered_1d =
  // MeltPoolDG::SimulationFactory<1,1>::registerSimulation("stefans_problem",
  // MeltPoolDG::makeDefaultSimulationFactoryFunction<SimulationStefansProblem<1>,1,1>() );
  // const static bool isRegistered_2d = SimulationFactory<2>::registerSimulation("stefans_problem"
  // makeDefaultSimulationFactoryFunction<class SimulationStefansProblem<2>>() );
  // const static bool isRegistered_3d = SimulationFactory<3>::registerSimulation("stefans_problem"
  // makeDefaultSimulationFactoryFunction<class SimulationStefansProblem<3>>() );
  /*
   * this function specifies the initial field of the level set equation
   */

} // namespace MeltPoolDG::Simulation::StefansProblem
