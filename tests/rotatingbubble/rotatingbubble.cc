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
// grid-specific libraries
#include <deal.II/distributed/tria.h>

namespace LevelSetParallel
{
  /*
   * this function specifies the initial field of the level set equation
   */
  
  template <int dim>
  class InitializePhi : public Function<dim>
  {
    public:
    InitializePhi()
      : Function<dim>(),
        epsInterface(0.0313)
    {}
    virtual double value( const Point<dim> & p,
                   const unsigned int component = 0) const
    {
    Point<2> center     = Point<2>(0.0,0.5); 
    const double radius = 0.25;

    return utilityFunctions::tanHyperbolicusCharacteristicFunction( 
           utilityFunctions::signedDistanceCircle( p, center, radius ), 
           this->epsInterface 
           );
    }

    void setEpsInterface(double eps){ this->epsInterface = eps; }

    double getEpsInterface(){ return this->epsInterface; }

    private:
      double epsInterface;
  };

  template <int dim>
  class ExactSolution : public Function<dim> 
  {
  public:
    double value(const Point<dim> &p,
                         const unsigned int /*component*/ = 0) const 
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
  class AdvectionField : public TensorFunction<1,dim> 
  {
    public:
      AdvectionField()
        : TensorFunction<1, dim>()
      {
      }

      Tensor<1, dim> value(const Point<dim> & p) const 
      {
        Tensor<1, dim> value_;
        
        const double x = p[0];
        const double y = p[1];
        
        value_[0] = 4*y;
        value_[1] = -4*x;
        
        return value_;
      }
  };

  template <int dim>
  double DirichletCondition<dim>::value(const Point<dim> & p, 
                                                const unsigned int /*component*/) const 
  {
    return -1.0;
  }

  /*
   *      This class collects all relevant input data for the level set simulation
   */

  template<int dim>
  class Simulation : public SimulationBase<dim>
  {
  public:
    Simulation() 
      : SimulationBase<dim>(MPI_COMM_WORLD)
    {
      set_parameters();
      set_field_conditions();
      create_spatial_discretization();
      set_boundary_conditions();
    }
    
    void set_parameters()
    {
      std::string paramfile;
      paramfile = "/home/schreter/deal/multiphaseflow/tests/rotatingbubble/rotatingbubble.prm";
      this->parameters.process_parameters_file(paramfile);
    }

    void set_boundary_conditions()
    {
      /*
       *  create a pair of (boundary_id, dirichlet_function)
       */
      const unsigned int inflow_bc = 1; // this should be replaced by an Enum
      //face->set_boundary_id( BoundaryTypesLevelSet::dirichlet_bc ); //@ does not work currently
      this->boundary_conditions.dirichlet_bc.emplace(std::make_pair(inflow_bc, std::make_shared<DirichletCondition<dim>>()));
      /*
       *  mark inflow edges with boundary label (no boundary on outflow edges must be prescribed
       *  due to the hyperbolic nature of the analyzed problem
       *
                  in      out
              -----------------
              |       :       |
        out   |       :       | in
              |_______________|
              |       :       |
        in    |       :       | out
              |       :       |
              -----------------
       *         out     in        
       */         
      for (auto &face : this->triangulation.active_face_iterators())
      if ( (face->at_boundary() ) ) 
      {
          face->set_boundary_id( inflow_bc );
          //const double half_line = (left_domain - right_domain) / 2;
          //if ( face->center()[0] == left_domain && face->center()[1]<half_line )
            //face->set_boundary_id( inflow_bc );
          //else if ( face->center()[0] == right_domain && face->center()[1]>half_line )
            //face->set_boundary_id( inflow_bc );
          //else if ( face->center()[1] == right_domain && face->center()[0]<half_line )
            //face->set_boundary_id( inflow_bc );
          //else if ( face->center()[1] == left_domain && face->center()[0]>half_line )
            //face->set_boundary_id( inflow_bc );
          //else
            //face->set_boundary_id( 42 );
      }  

    }

    void set_field_conditions()
    {   
        this->field_conditions.initial_field = std::make_shared<InitializePhi<dim>>(); 
        this->field_conditions.advection_field = std::make_shared<AdvectionField<dim>>(); 
    }

    void create_spatial_discretization()
    {
      GridGenerator::hyper_cube( this->triangulation, 
                                 left_domain, 
                                 right_domain );

      this->triangulation.refine_global( this->parameters.global_refinements );
    }
  
  private:
    double left_domain = -1.0;
    double right_domain = 1.0;

  };


  template class Simulation<2>; 
  //template class Simulation<3>; //@ does not work currently
} // LevelSetParallel


int main(int argc, char* argv[])
{

  using namespace LevelSetParallel;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  try
    {
      
      auto mySim = std::make_shared<Simulation<2>>();

      if ( mySim->parameters.dimension==2 )
      {
        mySim->parameters.print_parameters();
        const int degree = 2;
        LevelSetParallel::LevelSetEquation<2,degree> levelSet_equation_solver(
                                                        mySim
                                                     );
        levelSet_equation_solver.run() ;
        /*
         *  compute the error compared to the exact solution
         */
        //ExactSolution<2> sol;
        //sol.setEpsInterface(levelSet_equation_solver.epsilon);
        //levelSet_equation_solver.compute_error( sol ) ;
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

