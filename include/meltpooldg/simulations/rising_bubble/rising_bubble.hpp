#pragma once

#ifdef MELT_POOL_DG_WITH_ADAFLO
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
    namespace RisingBubble
    {

      using namespace dealii;

      template <int dim>
      class InitialValuesLS : public Function<dim>
      {
      public:
        InitialValuesLS ()
          :
          Function<dim>(1, 0)
        {}

        double value (const Point<dim> &p,
                      const unsigned int /*component*/) const
        {
          Point<dim>   center = dim == 2 ? Point<dim>(0.5, 0.5) : Point<dim>(0.5, 0.5, 0.5);
          const double radius = 0.25;
          return UtilityFunctions::CharacteristicFunctions::sgn(
            UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, center, radius));
        }
      };

      /*
       *      This class collects all relevant input data for the level set simulation
       */

      template <int dim>
      class SimulationRisingBubble : public SimulationBase<dim>
      {
      public:
        SimulationRisingBubble(std::string parameter_file, const MPI_Comm mpi_communicator)
          : SimulationBase<dim>(parameter_file, mpi_communicator)
        {
          this->set_parameters();
        }

        void
        create_spatial_discretization()
        {
          this->triangulation =
            std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);


          if constexpr(dim == 2)
          {
            // create mesh
            std::vector<unsigned int> subdivisions (dim, 5);
            subdivisions[dim-1] = 10;

            const Point<dim> bottom_left;
            const Point<dim> top_right   = (dim == 2 ?
                                            Point<dim>(1,2) :
                                            Point<dim>(1,1,2));
            GridGenerator::subdivided_hyper_rectangle (*this->triangulation,
                                                       subdivisions,
                                                       bottom_left,
                                                       top_right);
            
            // set boundary indicator to 2 on left and right face -> symmetry boundary
            typename parallel::distributed::Triangulation<dim>::active_cell_iterator
            cell = this->triangulation->begin(),
            endc = this->triangulation->end();

            for ( ; cell!=endc; ++cell)
              for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
                if (cell->face(face)->at_boundary() &&
                    (std::fabs(cell->face(face)->center()[0]-1)<1e-14 ||
                     std::fabs(cell->face(face)->center()[0])<1e-14))
                  cell->face(face)->set_boundary_id(2);

            this->triangulation->refine_global(this->parameters.base.global_refinements);
          }
          else
          {
            AssertThrow(false, ExcNotImplemented ());
          }

        }

        void
        set_boundary_conditions()
        {
           auto dirichlet = std::make_shared<Functions::ConstantFunction<dim>>(-1.0);

           // lower, right and left faces
           this->attach_no_slip_boundary_condition(0, "navier_stokes_u");
           // upper face
           this->attach_symmetry_boundary_condition(2, "navier_stokes_u");
           
           this->attach_dirichlet_boundary_condition(0, 
                                                     dirichlet,
                                                     "level_set"); 
           this->attach_dirichlet_boundary_condition(2, 
                                                     dirichlet,
                                                     "level_set"); 

           this->attach_fix_pressure_constant_condition(0, "navier_stokes_p");
        }

        void
        set_field_conditions()
        {
            this->attach_initial_condition( std::make_shared<InitialValuesLS<dim>>(),
                                            "level_set");
            this->attach_initial_condition( std::shared_ptr<Function<dim> >(new Functions::ZeroFunction<dim>(dim)),
                                            "navier_stokes_u");
        }
      };

    } // namespace RisingBubble
  }   // namespace Simulation
} // namespace MeltPoolDG

#endif
