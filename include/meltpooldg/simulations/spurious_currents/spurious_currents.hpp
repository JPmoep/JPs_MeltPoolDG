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
        InitializePhi()
          : Function<dim>()
        {}
        virtual double
        value(const Point<dim> &p, const unsigned int component = 0) const
        {    
            (void) component;
            
            // set radius of bubble to 0.5, slightly shifted away from the center
            Point<dim> center;
            for (unsigned int d=0; d<dim; ++d)
              center[d] = 0.02+0.01*d;
            return p.distance(center) - 0.5;
        }
      };

      /* for constant Dirichlet conditions we could also use the ConstantFunction
       * utility from dealii
       */
      template <int dim>
      class DirichletCondition : public Function<dim>
      {
      public:
        DirichletCondition()
          : Function<dim>()
        {}

        double
        value(const Point<dim> &p, const unsigned int component = 0) const
        {
          (void)p;
          (void)component;
          return -1.0;
        }
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
        create_spatial_discretization()
        {
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);

          if constexpr (dim == 2)
            {
              GridGenerator::subdivided_hyper_cube (*this->triangulation, 20 /*TODO*/, -2.5, 2.5);
            }
          else
            {
              AssertThrow(false, ExcNotImplemented());
            }
        }

        void
        set_boundary_conditions()
        {
          auto dirichlet = std::make_shared<DirichletCondition<dim>>();
          this->attach_dirichlet_boundary_condition(0, dirichlet, "level_set");
          
          this->attach_no_slip_boundary_condition(0, "navier_stokes");
          this->attach_fix_pressure_constant_condition(0, "navier_stokes");

        }

        void
        set_field_conditions()
        {
          this->field_conditions.initial_field = std::make_shared<InitializePhi<dim>>();
          
          this->attach_initial_condition(
            std::make_shared<Functions::ZeroFunction<dim>>(dim), "navier_stokes");
        }
      };

    } // namespace SpuriousCurrents
  }   // namespace Simulation
} // namespace MeltPoolDG
