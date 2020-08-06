#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>

#include <iostream>
#include "levelset.hpp"
#include "levelsetparameters.hpp"
#include "utilityFunctions.hpp"
#include <cmath>


template <int dim>
double InitializePhi<dim>::value(const Point<dim> &p,
                                 const unsigned int /*component*/) const
{
    Point<2> center     = Point<2>(0,0); 
    const double radius = 0.2;
    using namespace utilityFunctions;
    return tanHyperbolicusCharacteristicFunction( 
                signedDistanceCircle( p, center, radius ), epsInterface );
}

// specify advection field
template <int dim>
Tensor<1, dim> AdvectionField<dim>::value(const Point<dim> & p) const 
{
  //const double t = this->get_time();
  
  const std::vector advectionConstant    = {0.1,0.0,0.0};  // (m/s)
  Tensor<1, dim> value_;
  
  value_[0] = advectionConstant[0];
  value_[1] = advectionConstant[1];
  
  return value_;
}

LevelSetParameters params {
    .timeStep                  = 0.01,   
    .maxTime                   = dealii::numbers::PI/2,  
    .theta                     = 0.5, // timestepping; 0=explicit; 0.5=trapez; 1.0=implicit
    .diffusivity               = 0.0, // artificial diffusivity
    .activateReinitialization  = true,
    .computeVolume             = true,
    .dirichletBoundaryValue    = -1.0,  
    .levelSetDegree            = 2,  
    .characteristicMeshSize    = 0.0,  
    .epsInterface              = 0.0,
};

int main()
{

  try
    {

      const int nDim = 2;
      const double leftDomain   = -0.5;            // (m)
      const double rightDomain  = 0.5;
      const int nMeshRefinements = 6; 
      
      params.characteristicMeshSize  = (rightDomain-leftDomain) / ( std::pow(2,nMeshRefinements )); 
      params.epsInterface = 2.0* params.characteristicMeshSize;
      
      Triangulation<nDim>        triangulation;
      GridGenerator::hyper_cube( triangulation, 
                                 leftDomain, 
                                 rightDomain );

      // mark dirichlet edges
      for (auto &face : triangulation.active_face_iterators())
        if ( (face->at_boundary() ) ) // && (face->center()[0] == InputParameters::leftDomain) )
        {
            face->set_boundary_id ( utilityFunctions::BCTypes::dirichlet);
        }

      triangulation.refine_global( nMeshRefinements );

      std::cout << "Input: " << nDim << std::endl;
      LevelSet::LevelSetEquation<nDim> levelSet_equation_solver(
              params,
              triangulation
              );
    
    AdvectionField<nDim> adv;
    InitializePhi<nDim> ini;
    ini.setEpsInterface(params.epsInterface);
    
    levelSet_equation_solver.run( ini, adv ) ;
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


