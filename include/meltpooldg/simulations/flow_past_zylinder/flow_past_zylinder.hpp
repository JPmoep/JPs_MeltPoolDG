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
    namespace FlowPastZylinder
    {
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
      class SimulationFlowPastZylinder : public SimulationBase<dim>
      {
      public:
        SimulationFlowPastZylinder(std::string parameter_file, const MPI_Comm mpi_communicator)
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
  Triangulation<2> left, middle, right, tmp, tmp2;
  GridGenerator::subdivided_hyper_rectangle(left, std::vector<unsigned int>({3U, 4U}),
                                            Point<2>(), Point<2>(0.3, 0.41), false);
  GridGenerator::subdivided_hyper_rectangle(right, std::vector<unsigned int>({18U, 4U}),
                                            Point<2>(0.7, 0), Point<2>(2.5, 0.41), false);

  // create middle part first as a hyper shell
  GridGenerator::hyper_shell(middle, Point<2>(0.5, 0.2), 0.05, 0.2, 4, true);
  middle.reset_all_manifolds();
  for (const auto &cell : middle.active_cell_iterators())
    for (unsigned int f=0; f<GeometryInfo<2>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary() &&
          Point<2>(0.5,0.2).distance(cell->face(f)->center())<=0.05)
        cell->face(f)->set_manifold_id(0);

  middle.set_manifold(0, PolarManifold<2>(Point<2>(0.5, 0.2)));
  middle.refine_global(1);

  // then move the vertices to the points where we want them to be to create a
  // slightly asymmetric cube with a hole
  for (Triangulation<2>::cell_iterator cell = middle.begin();
       cell != middle.end(); ++cell)
    for (unsigned int v=0; v < GeometryInfo<2>::vertices_per_cell; ++v)
      {
        Point<2> &vertex = cell->vertex(v);
        if (std::abs(vertex[0] - 0.7) < 1e-10 &&
            std::abs(vertex[1] - 0.2) < 1e-10)
          vertex = Point<2>(0.7, 0.205);
        else if (std::abs(vertex[0] - 0.6) < 1e-10 &&
                 std::abs(vertex[1] - 0.3) < 1e-10)
          vertex = Point<2>(0.7, 0.41);
        else if (std::abs(vertex[0] - 0.6) < 1e-10 &&
                 std::abs(vertex[1] - 0.1) < 1e-10)
          vertex = Point<2>(0.7, 0);
        else if (std::abs(vertex[0] - 0.5) < 1e-10 &&
                 std::abs(vertex[1] - 0.4) < 1e-10)
          vertex = Point<2>(0.5, 0.41);
        else if (std::abs(vertex[0] - 0.5) < 1e-10 &&
                 std::abs(vertex[1] - 0.0) < 1e-10)
          vertex = Point<2>(0.5, 0.0);
        else if (std::abs(vertex[0] - 0.4) < 1e-10 &&
                 std::abs(vertex[1] - 0.3) < 1e-10)
          vertex = Point<2>(0.3, 0.41);
        else if (std::abs(vertex[0] - 0.4) < 1e-10 &&
                 std::abs(vertex[1] - 0.1) < 1e-10)
          vertex = Point<2>(0.3, 0);
        else if (std::abs(vertex[0] - 0.3) < 1e-10 &&
                 std::abs(vertex[1] - 0.2) < 1e-10)
          vertex = Point<2>(0.3, 0.205);
        else if (std::abs(vertex[0] - 0.56379) < 1e-4 &&
                 std::abs(vertex[1] - 0.13621) < 1e-4)
          vertex = Point<2>(0.59, 0.11);
        else if (std::abs(vertex[0] - 0.56379) < 1e-4 &&
                 std::abs(vertex[1] - 0.26379) < 1e-4)
          vertex = Point<2>(0.59, 0.29);
        else if (std::abs(vertex[0] - 0.43621) < 1e-4 &&
                 std::abs(vertex[1] - 0.13621) < 1e-4)
          vertex = Point<2>(0.41, 0.11);
        else if (std::abs(vertex[0] - 0.43621) < 1e-4 &&
                 std::abs(vertex[1] - 0.26379) < 1e-4)
          vertex = Point<2>(0.41, 0.29);
      }

  // refine once to create the same level of refinement as in the
  // neighboring domains
  middle.refine_global(1);

  // must copy the triangulation because we cannot merge triangulations with
  // refinement...
  GridGenerator::flatten_triangulation(middle, tmp2);

  if (dim == 2)
    GridGenerator::merge_triangulations (tmp2, right, *this->triangulation);
  else
    {
      GridGenerator::merge_triangulations (left, tmp2, tmp);
      GridGenerator::merge_triangulations (tmp, right, *this->triangulation);
    }

  // Set the left boundary (inflow) to 1, the right to 2, the rest to 0.
  for (Triangulation<2>::active_cell_iterator cell=this->triangulation->begin() ;
       cell != this->triangulation->end(); ++cell)
    for (unsigned int f=0; f<GeometryInfo<2>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary())
        {
          if (std::abs(cell->face(f)->center()[0] - (dim == 2 ? 0.3 : 0)) < 1e-12)
            cell->face(f)->set_all_boundary_ids(1);
          else if (std::abs(cell->face(f)->center()[0]-2.5) < 1e-12)
            cell->face(f)->set_all_boundary_ids(2);
          else if (Point<2>(0.5,0.2).distance(cell->face(f)->center())<=0.05)
            {
              cell->face(f)->set_all_manifold_ids(10);
              cell->face(f)->set_all_boundary_ids(0);
            }
          else
            cell->face(f)->set_all_boundary_ids(0);
        }
  this->triangulation->set_manifold(10, PolarManifold<2>(Point<2>(0.5,0.2)));
          }
          else
          {
            AssertThrow(false, ExcNotImplemented ());
          }

        }

        void
        set_boundary_conditions()
        {
        }

        void
        set_field_conditions()
        {
        }
      };

    } // namespace FlowPastZylinder
  }   // namespace Simulation
} // namespace MeltPoolDG
