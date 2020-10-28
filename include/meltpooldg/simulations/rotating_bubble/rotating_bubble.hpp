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
    namespace RotatingBubble
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
          , epsInterface(0.0313)
        {}
        virtual double
        value(const Point<dim> &p, const unsigned int component = 0) const
        {
          (void)component;

          Point<dim>   center = dim == 1 ? Point<dim>(0.0) : Point<dim>(0.0, 0.5);
          const double radius = 0.25;

          return UtilityFunctions::CharacteristicFunctions::sgn(
            UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, center, radius));

          /*
           *  Alternatively, a tanh function could be used, corresponding to the
           *  analytic solution of the reinitialization problem
           */
          // return UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function(
          // UtilityFunctions::DistanceFunctions::spherical_manifold<dim>( p, center, radius ),
          // this->epsInterface
          //);
        }

        void
        setEpsInterface(double eps)
        {
          this->epsInterface = eps;
        }

        double
        getEpsInterface()
        {
          return this->epsInterface;
        }

      private:
        double epsInterface;
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
          Point<dim>   center = dim == 1 ? Point<dim>(0.0) : Point<dim>(0.0, 0.5);
          const double radius = 0.25;

          return UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function(
            UtilityFunctions::DistanceFunctions::spherical_manifold(p, center, radius),
            eps_interface);
        }

      private:
        double eps_interface;
      };


      template <int dim>
      class AdvectionField : public TensorFunction<1, dim>
      {
      public:
        AdvectionField()
          : TensorFunction<1, dim>()
        {}

        Tensor<1, dim>
        value(const Point<dim> &p) const
        {
          Tensor<1, dim> value_;

          if constexpr (dim == 2)
            {
              const double x = p[0];
              const double y = p[1];

              value_[0] = 4 * y;
              value_[1] = -4 * x;
            }
          else
            AssertThrow(false, ExcMessage("Advection field for dim!=2 not implemented"));

          return value_;
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
      class SimulationRotatingBubble : public SimulationBase<dim>
      {
      public:
        SimulationRotatingBubble(std::string parameter_file, const MPI_Comm mpi_communicator)
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
        {
          /*
           *  create a pair of (boundary_id, dirichlet_function)
           */
          constexpr types::boundary_id inflow_bc  = 42;
          constexpr types::boundary_id do_nothing = 0;

          this->boundary_conditions.dirichlet_bc.emplace(
            std::make_pair(inflow_bc, std::make_shared<DirichletCondition<dim>>()));
          /*
           *  mark inflow edges with boundary label (no boundary on outflow edges must be prescribed
           *  due to the hyperbolic nature of the analyzed problem
           *
                      out    in
          (-1,1)  +---------------+ (1,1)
                  |       :       |
            in    |       :       | out
                  |_______________|
                  |       :       |
            out   |       :       | in
                  |       :       |
                  +---------------+
           * (-1,-1)  in     out   (1,-1)
           */
          if constexpr (dim == 2)
            {
              for (auto &face : this->triangulation->active_face_iterators())
                if ((face->at_boundary()))
                  {
                    const double half_line = (right_domain + left_domain) / 2;

                    if (face->center()[0] == left_domain && face->center()[1] > half_line)
                      face->set_boundary_id(inflow_bc);
                    else if (face->center()[0] == right_domain && face->center()[1] < half_line)
                      face->set_boundary_id(inflow_bc);
                    else if (face->center()[1] == right_domain && face->center()[0] > half_line)
                      face->set_boundary_id(inflow_bc);
                    else if (face->center()[1] == left_domain && face->center()[0] < half_line)
                      face->set_boundary_id(inflow_bc);
                    else
                      face->set_boundary_id(do_nothing);
                  }
            }
          else
            {
              (void)do_nothing; // suppress unused variable for 1D
            }
        }

        void
        set_field_conditions()
        {
          this->field_conditions.initial_field        = std::make_shared<InitializePhi<dim>>();
          this->field_conditions.advection_field      = std::make_shared<AdvectionField<dim>>();
          this->field_conditions.exact_solution_field = std::make_shared<ExactSolution<dim>>(0.01);
        }

      private:
        double left_domain  = -1.0;
        double right_domain = 1.0;
      };

    } // namespace RotatingBubble
  }   // namespace Simulation
} // namespace MeltPoolDG
