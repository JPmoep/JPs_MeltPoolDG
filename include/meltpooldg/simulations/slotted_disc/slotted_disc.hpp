// deal-specific libraries
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/distributed/tria.h>
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
    namespace SlottedDisc
    {
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
          // default distances
          double d_AB       = std::numeric_limits<double>::max();
          double d_BC       = std::numeric_limits<double>::max();
          double d_CD       = std::numeric_limits<double>::max();
          double d_manifold = std::numeric_limits<double>::max();
          double d_min;
          // geometric values
          const double radius  = 0.3;
          const double delta_y = 0.04196;
          Point<dim>   center  = dim == 1 ? Point<dim>(0.0) : Point<dim>(0.0, 0.5);
          Point<dim>   pA      = dim == 1 ? Point<dim>(0.0) : Point<dim>(-0.05, 0.2 + delta_y);
          Point<dim>   pB      = dim == 1 ? Point<dim>(0.0) : Point<dim>(-0.05, 0.7);
          Point<dim>   pC      = dim == 1 ? Point<dim>(0.0) : Point<dim>(0.05, 0.7);
          Point<dim>   pD      = dim == 1 ? Point<dim>(0.0) : Point<dim>(0.05, 0.2 + delta_y);

          if (p[1] <= pA[1])
            {
              if (p[0] >= pA[0] && p[0] <= pD[0])
                { // region 10 and 11
                  d_AB  = UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, pA, 0.0);
                  d_CD  = UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, pD, 0.0);
                  d_min = std::max(d_AB, d_CD);
                }
              else
                { // boundary region of 10 and 11
                  d_AB = UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, pA, 0.0);
                  d_CD = UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, pD, 0.0);
                  d_manifold =
                    UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, center, radius);
                  d_min = std::max(d_AB, d_CD);
                  d_min = std::max(d_manifold, d_min);
                }
            }
          else if (p[1] >= pB[1])
            {
              if (p[0] <= pB[0])
                { // region 3
                  d_BC = std::abs(
                    UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, pB, 0.0));
                  d_manifold =
                    UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, center, radius);
                  d_min = std::min(d_BC, d_manifold);
                }
              else if (p[0] >= pC[0])
                { // region 4
                  d_BC = std::abs(
                    UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, pC, 0.0));
                  d_manifold =
                    UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, center, radius);
                  d_min = std::min(d_BC, d_manifold);
                }
              else if (p[0] > pB[0] && p[0] < pC[0])
                { // region 2
                  d_BC = UtilityFunctions::DistanceFunctions::infinte_line<dim>(p, pB, pC);
                  d_manifold =
                    UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, center, radius);
                  d_min = std::min(d_BC, d_manifold);
                }
            }
          else if (p[0] > center[0] - radius && p[0] < center[0] + radius) // region 1, 5-7, 8, 9
            {
              if (p[0] > pB[0] && p[0] < pC[0]) // region 5-7
                {
                  d_AB  = -UtilityFunctions::DistanceFunctions::infinte_line<dim>(p, pA, pB);
                  d_BC  = -UtilityFunctions::DistanceFunctions::infinte_line<dim>(p, pB, pC);
                  d_CD  = -UtilityFunctions::DistanceFunctions::infinte_line<dim>(p, pC, pD);
                  d_min = std::max(d_AB, d_BC);
                  d_min = std::max(d_CD, d_min);
                }
              else
                {
                  d_AB = UtilityFunctions::DistanceFunctions::infinte_line<dim>(p, pA, pB);
                  d_CD = UtilityFunctions::DistanceFunctions::infinte_line<dim>(p, pC, pD);
                  d_manifold =
                    UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, center, radius);
                  d_min = std::min(d_AB, d_CD);
                  d_min = std::min(d_min, d_manifold);
                }
            }
          else
            { // outer region
              d_min =
                UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, center, radius);
            }

          // return the smallest distance
          return UtilityFunctions::CharacteristicFunctions::sgn(d_min);
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
          Point<dim>   center  = dim == 1 ? Point<dim>(0.0) : Point<dim>(0.0, 0.5);
          const double radius  = 0.3;
          const double delta_y = 0.04196;
          Point<dim>   pA      = dim == 1 ? Point<dim>(0.0) : Point<dim>(-0.05, 0.2 + delta_y);
          Point<dim>   pB      = dim == 1 ? Point<dim>(0.0) : Point<dim>(-0.05, 0.7);
          Point<dim>   pC      = dim == 1 ? Point<dim>(0.0) : Point<dim>(0.05, 0.7);
          Point<dim>   pD      = dim == 1 ? Point<dim>(0.0) : Point<dim>(0.05, 0.2 + delta_y);

          // default distance
          double d_AB       = std::numeric_limits<double>::max();
          double d_BC       = std::numeric_limits<double>::max();
          double d_CD       = std::numeric_limits<double>::max();
          double d_manifold = std::numeric_limits<double>::max();
          double d_min      = std::numeric_limits<double>::max();

          if (p[1] <= pA[1])
            {
              if (p[0] >= pA[0] && p[0] <= pD[0])
                { // region 10 and 11
                  d_AB  = UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, pA, 0.0);
                  d_CD  = UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, pD, 0.0);
                  d_min = std::min(d_AB, d_CD);
                }
              else
                { // boundary region of 10 and 11
                  d_AB = UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, pA, 0.0);
                  d_CD = UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, pD, 0.0);
                  d_manifold =
                    UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, center, radius);
                  d_min = std::min(d_AB, d_CD);
                  d_min = std::min(d_manifold, d_min);
                }
            }
          else if (p[1] >= pB[1])
            {
              if (p[0] <= pB[0])
                { // region 3
                  d_BC = UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, pB, 0.0);
                  d_manifold =
                    UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, center, radius);
                  d_min = std::min(d_BC, d_manifold);
                }
              else if (p[0] >= pC[0])
                { // region 4
                  d_BC = UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, pC, 0.0);
                  d_manifold =
                    UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, center, radius);
                  d_min = std::min(d_BC, d_manifold);
                }
              else if (p[0] > pB[0] && p[0] < pC[0])
                { // region 2
                  d_BC = UtilityFunctions::DistanceFunctions::infinte_line<dim>(p, pB, pC);
                  d_manifold =
                    UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, center, radius);
                  d_min = std::min(d_BC, d_manifold);
                }
            }
          else if (p[0] > center[0] - radius && p[0] < center[0] + radius) // region 1, 5-7, 8, 9
            {
              if (p[0] > pB[0] && p[0] < pC[0]) // region 5-7
                {
                  d_AB  = UtilityFunctions::DistanceFunctions::infinte_line<dim>(p, pA, pB);
                  d_BC  = UtilityFunctions::DistanceFunctions::infinte_line<dim>(p, pB, pC);
                  d_CD  = UtilityFunctions::DistanceFunctions::infinte_line<dim>(p, pC, pD);
                  d_min = std::min(d_AB, d_BC);
                  d_min = std::min(d_CD, d_min);
                }
              else
                {
                  d_AB = UtilityFunctions::DistanceFunctions::infinte_line<dim>(p, pA, pB);
                  d_CD = UtilityFunctions::DistanceFunctions::infinte_line<dim>(p, pC, pD);
                  d_manifold =
                    UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, center, radius);
                  d_min = std::min(d_AB, d_CD);
                  d_min = std::min(d_min, d_manifold);
                }
            }
          else
            { // outer region
              d_min =
                UtilityFunctions::DistanceFunctions::spherical_manifold<dim>(p, center, radius);
            }

          // return the smallest distance
          return UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function(
            d_min, eps_interface);
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
              const double x  = p[0];
              const double y  = p[1];

              value_[0] = 4*y;
              value_[1] = -4*x;
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
         {
         }

         double value(const Point<dim> &p,
                              const unsigned int component = 0) const
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
      class SimulationSlottedDisc : public SimulationBase<dim>
      {
      public:
        SimulationSlottedDisc(std::string parameter_file, const MPI_Comm mpi_communicator)
          : SimulationBase<dim>(parameter_file, mpi_communicator)
        {
          this->set_parameters();
        }

        void
        create_spatial_discretization()
        {
          if (dim == 1)
            {
              AssertDimension(Utilities::MPI::n_mpi_processes(this->mpi_communicator), 1);
              this->triangulation = std::make_shared<Triangulation<dim>>();
            }
          else
            this->triangulation =
              std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
          GridGenerator::hyper_cube(*this->triangulation, left_domain, right_domain);
          this->triangulation->refine_global(this->parameters.base.global_refinements);
        }

        void
        set_boundary_conditions()
        {
          /*
           *  create a pair of (boundary_id, dirichlet_function)
           */
          constexpr types::boundary_id inflow_bc  = 42;
          constexpr types::boundary_id do_nothing = 0;

          this->attach_dirichlet_boundary_condition(inflow_bc,
                                                    std::make_shared<DirichletCondition<dim>>(),
                                                    this->parameters.base.problem_name);
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
          this->attach_initial_condition(std::make_shared<InitializePhi<dim>>(), "level_set");
          this->attach_advection_field(std::make_shared<AdvectionField<dim>>(), "level_set");
          this->attach_exact_solution(std::make_shared<ExactSolution<dim>>(0.01), "level_set");
        }

      private:
        double left_domain  = -1.0;
        double right_domain = 1.0;
      };
    } // namespace SlottedDisc
  }   // namespace Simulation
} // namespace MeltPoolDG

