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
          (void)component;

          Point<dim>   center = dim == 1 ? Point<dim>(0.0) : Point<dim>(1.3, 0.2);
          const double radius = 0.1;

          return UtilityFunctions::CharacteristicFunctions::sgn(
            UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, center, radius));
        }
      };

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

        virtual void vector_value(const Point<dim>     &p,
                                        Vector<double> &values) const
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
            Point<2> p1(0,0);
            Point<2> p2(2.5, 0.4);
            std::vector<unsigned int> refinements({50, 8});
            Triangulation<2> tmp;
            GridGenerator::subdivided_hyper_rectangle(tmp, refinements, p1, p2);
            std::set<Triangulation<2>::active_cell_iterator> cells_in_void;
            for (Triangulation<2>::active_cell_iterator cell = tmp.begin();
                cell != tmp.end(); ++cell)
              if (cell->center()[0] > 0.45 && cell->center()[0]<0.55 &&
                  cell->center()[1] > 0.15 && cell->center()[1]<0.25)
                cells_in_void.insert(cell);
            GridGenerator::create_triangulation_with_removed_cells(tmp, cells_in_void, *this->triangulation);

            // shift cells at the upper end of the domain from 0.40 to 0.41. It
            // corresponds to faces with id 3
            for (Triangulation<2>::cell_iterator cell = this->triangulation->begin();
                cell != this->triangulation->end(); ++cell)
              if (cell->at_boundary(3) && cell->face(3)->center()[1] > 0.39999999999)
                for (unsigned int v=0; v<GeometryInfo<2>::vertices_per_face; ++v)
                  cell->face(3)->vertex(v)[1] = 0.41;

            // Set the left boundary (inflow) to 1, the right to 2, the rest to 0.
            for (Triangulation<2>::active_cell_iterator cell=this->triangulation->begin() ;
                cell != this->triangulation->end(); ++cell)
              for (unsigned int f=0; f<GeometryInfo<2>::faces_per_cell; ++f)
                if (cell->face(f)->at_boundary())
                  {
                    if (std::abs(cell->face(f)->center()[0]) < 1e-12)
                      cell->face(f)->set_all_boundary_ids(1);
                    else if (std::abs(cell->face(f)->center()[0]-2.5) < 1e-12)
                      cell->face(f)->set_all_boundary_ids(2);
                    else
                      cell->face(f)->set_all_boundary_ids(0);
                  }
          }
          else
          {
            AssertThrow(false, ExcNotImplemented ());
          }

        }

        void
        set_boundary_conditions()
        {
          
           auto dirichlet = std::make_shared<DirichletCondition<dim>>();

           this->attach_dirichlet_boundary_condition(0, 
                                                     dirichlet,
                                                     "level_set"); 
           this->attach_dirichlet_boundary_condition(1, 
                                                     dirichlet,
                                                    "level_set");

           this->attach_no_slip_boundary_condition(0, "navier_stokes_u");

           this->attach_dirichlet_boundary_condition(1, 
                                     std::shared_ptr<Function<dim> >(new InflowVelocity<dim>(0., true)),
                                     "navier_stokes_u");
           
           this->attach_neumann_boundary_condition(2, 
                                                   std::shared_ptr<Function<dim> > (new Functions::ZeroFunction<dim>(1)),
                                                   "navier_stokes_p");
        }

        void
        set_field_conditions()
        {
            this->field_conditions.initial_field = std::make_shared<InitializePhi<dim>>();
            this->attach_initial_condition( std::shared_ptr<Function<dim> >(new InflowVelocity<dim>(0., false)),
                                            "navier_stokes_u");
        }
      };

    } // namespace FlowPastZylinder
  }   // namespace Simulation
} // namespace MeltPoolDG
