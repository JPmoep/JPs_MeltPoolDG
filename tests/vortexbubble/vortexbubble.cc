#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>
#include <iostream>
#include "levelsetParallel.hpp"
#include "levelsetparameters.hpp"
#include "utilityFunctions.hpp"
#include "simulationbase.hpp"
#include <cmath>

namespace LevelSetParallel
{

template <int dim>
class ExactSolution : public Function<dim> 
{
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int /*component*/ = 0) const override
  {
    InitializePhi<dim> ini;

    ini.setEpsInterface(epsInterface);
    return ini.value(p);
  }

  void setEpsInterface(double eps){ this->epsInterface = eps; }

  double getEpsInterface(){ return this->epsInterface; }

  private:
        double epsInterface;
};

template <int dim>
double InitializePhi<dim>::value(const Point<dim> &p,
                                 const unsigned int /*component*/) const
{
    Point<dim> center     = Point<dim>(0.5,0.75); 
    const double radius = 0.15;

    return utilityFunctions::tanHyperbolicusCharacteristicFunction( 
           utilityFunctions::signedDistanceCircle( p, center, radius ), 
           this->epsInterface 
           );

}

// specify advection field
template <int dim>
Tensor<1, dim> AdvectionField<dim>::value(const Point<dim> & p) const 
{
  const double time = this->get_time();

  Point<dim> value;
  const double Tf = 2.0;
  const double x = p[0];
  const double y = p[1];
  
  const double reverseCoefficient = std::cos(numbers::PI * time / Tf );

  value[0] = reverseCoefficient * (  std::sin( 2. * numbers::PI * y) * std::pow( std::sin( numbers::PI * x ), 2.) );
  value[1] = reverseCoefficient * ( -std::sin( 2. * numbers::PI * x) * std::pow( std::sin( numbers::PI * y ), 2.) );
  return value;
}

template <int dim>
double DirichletCondition<dim>::value(const Point<dim> & p, 
                                              const unsigned int /*component*/) const 
{
  return -1.0;
}

// specify field
template <int dim>
void DirichletCondition<dim>::markDirichletEdges(Triangulation<dim>& triangulation) const 
{
  //for (auto &face : triangulation.active_face_iterators())
    //if ( (face->at_boundary() ) ) //&& (face->center()[0] == InputParameters::leftDomain) )
    //{
        //face->set_boundary_id ( utilityFunctions::BoundaryConditions::Types::dirichlet);
    //}
}

template<int dim>
class Simulation : public SimulationBase<dim>
{
public:
    Simulation() : SimulationBase<dim>()
    {}

    void set_parameters()
    {
    }

    void set_boundary_conditions()
    {
        this->boundary_conditions.dirichlet_bc.emplace(std::make_pair(utilityFunctions::BoundaryConditions::Types::dirichlet, std::make_shared<DirichletCondition<dim>>()));
    }

    void set_field_conditions()
    {   
        this->field_conditions.initial_field = std::make_shared<InitializePhi<dim>>(); 
        //// @ how to set correct time of Advection field?
        this->field_conditions.advection_field = std::make_shared<AdvectionField<dim>>(); 
    }

    void create_spatial_discretization()
    {
    }

};

template class Simulation<2>; 
//template class Simulation<3>; //@ does not work currently
} // LevelSetParallel



int main(int argc, char* argv[])
{

  using namespace LevelSetParallel;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPI_Comm mpi_communicator;
  mpi_communicator = MPI_COMM_WORLD;
  
  ConditionalOStream pcout(std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator) == 0));

  try
    {
      const double leftDomain    = 0.0;            // (m)
      const double rightDomain   = 1.0;
 
      std::string paramfile;
      paramfile = "/home/schreter/deal/multiphaseflow/tests/vortexbubble/vortexbubble.prm";

      LevelSetParameters parameters (paramfile);
      
      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
          parameters.print_parameters();

      if ( parameters.dimension==2 )
      {
          parallel::distributed::Triangulation<2>        triangulation(mpi_communicator);
          GridGenerator::hyper_cube( triangulation, 
                                     leftDomain, 
                                     rightDomain );

          triangulation.refine_global( parameters.global_refinements );

          AdvectionField<2> adv;

          LevelSetParallel::LevelSetEquation<2> levelSet_equation_solver(
                  parameters,
                  triangulation,
                  adv,
                  mpi_communicator
                  );

        InitializePhi<2> ini;
        ini.setEpsInterface(levelSet_equation_solver.epsilon);

        DirichletCondition<2> dirichlet;
        dirichlet.markDirichletEdges( triangulation );

        levelSet_equation_solver.run( ini, dirichlet ) ;
        ExactSolution<2> sol;
        sol.setEpsInterface(levelSet_equation_solver.epsilon);
        levelSet_equation_solver.compute_error( sol ) ;

      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}


