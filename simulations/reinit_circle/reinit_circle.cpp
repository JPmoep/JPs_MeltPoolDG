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
#include <meltpooldg/utilities/utilityfunctions.hpp>
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/interface/problemselector.hpp>

namespace MeltPoolDG
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
    (void)component;
    Point<2> center     = Point<2>(0.0,0.5); 
    const double radius = 0.25;
    //return UtilityFunctions::CharacteristicFunctions::sgn( 
                //UtilityFunctions::DistanceFunctions::spherical_manifold<dim>( p, center, radius ));

    /*
     *  Alternatively, a tanh function could be used, corresponding to the
     *  analytic solution of the reinitialization problem
     */
    return UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function( 
           UtilityFunctions::DistanceFunctions::spherical_manifold<dim>( p, center, radius ), 
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
    ExactSolution(const double eps)
    : Function<dim>(),
    eps_interface(eps)
    {
    }

    double value(const Point<dim> &p,
                         const unsigned int component = 0) const 
    {
      (void)component;
      Point<2> center     = Point<2>(0.0,0.5); 
      const double radius = 0.25;
      return UtilityFunctions::CharacteristicFunctions::tanh_characteristic_function( 
             UtilityFunctions::DistanceFunctions::spherical_manifold( p, center, radius ), 
             eps_interface 
             );
    }
    private:
      double eps_interface;
  };
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
    }
    
    void set_parameters()
    {
      std::string paramfile;
      paramfile = "reinit_circle.json";
      this->parameters.process_parameters_file(paramfile);
    }

    void create_spatial_discretization()
    {
      GridGenerator::hyper_cube( this->triangulation, 
                                 left_domain, 
                                 right_domain );

      this->triangulation.refine_global( this->parameters.global_refinements );
      /* 
       * alternative with shared pointer
       * */
      this->triangulation_shared = std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator);
      GridGenerator::hyper_cube( *this->triangulation_shared, 
                                 left_domain, 
                                 right_domain );
      this->triangulation_shared->refine_global( this->parameters.global_refinements );
    }

    void set_boundary_conditions()
    {
    }

    void set_field_conditions()
    {   
        this->field_conditions.initial_field =        std::make_shared<InitializePhi<dim>>(); 
        this->field_conditions.exact_solution_field = std::make_shared<ExactSolution<dim>>(0.01);
            //GridTools::minimal_cell_diameter(this->triangulation)/(std::sqrt(dim)*2)); 
    }
  
  private:
    double left_domain = -1.0;
    double right_domain = 1.0;

  };

  //template class Simulation<3>; //@ does not work currently
template class Simulation<2>; 


} // namespace MeltPoolDG

int main(int argc, char* argv[])
{
  using namespace dealii;
  using namespace MeltPoolDG;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  
  try
    {
      // @ todo: incorporate better way for template parameter degree
      const int degree = 1;

      auto sim = std::make_shared<Simulation<2>>();

      if ( sim->parameters.dimension==2 )
      {
        sim->create();
        auto problem = ProblemSelector<2,degree>::get_problem(sim);
        problem->run();
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


