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
      paramfile = "rotatingbubble.json";
      this->parameters.process_parameters_file(paramfile);

      /*
       * @todo (wip)
       * new parameter handling
       */
      
           //dealii::ParameterHandler prm;

      //std::string file_name;
      //file_name = "rotatingbubble_newinterface.prm";
      //this->ls_parameters.process_parameters_file(file_name);

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
      /*
       *  create a pair of (boundary_id, dirichlet_function)
       */
      const unsigned int inflow_bc = 1; // @  this should be replaced by an Enum
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
          const double half_line = (left_domain - right_domain) / 2;
          if ( face->center()[0] == left_domain && face->center()[1]>half_line )
            face->set_boundary_id( inflow_bc );
          else if ( face->center()[0] == right_domain && face->center()[1]<half_line )
            face->set_boundary_id( inflow_bc );
          else if ( face->center()[1] == right_domain && face->center()[0]>half_line )
            face->set_boundary_id( inflow_bc );
          else if ( face->center()[1] == left_domain && face->center()[0]<half_line )
            face->set_boundary_id( inflow_bc );
          else
            face->set_boundary_id( inflow_bc );
      }  

    }

    void set_field_conditions()
    {   
        this->field_conditions.initial_field =        std::make_shared<InitializePhi<dim>>(); 
        this->field_conditions.advection_field =      std::make_shared<AdvectionField<dim>>(); 
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


