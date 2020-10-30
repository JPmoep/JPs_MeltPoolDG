#pragma once
// dealii
#include <deal.II/distributed/tria.h>
#include <deal.II/base/exceptions.h>
// MeltPoolDG
#include <meltpooldg/interface/boundaryconditions.hpp>
#include <meltpooldg/interface/fieldconditions.hpp>
#include <meltpooldg/interface/parameters.hpp>
// c++
#include <memory>

namespace MeltPoolDG
{
  using namespace dealii;

  template <int dim, int spacedim = dim>
  class SimulationBase
  {
  public:
    SimulationBase(std::string parameter_file_in, MPI_Comm mpi_communicator_in)
      : parameter_file(parameter_file_in)
      , mpi_communicator(mpi_communicator_in)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      set_parameters();
    }

    virtual ~SimulationBase() = default;

    virtual void
    set_parameters()
    {
      this->parameters.process_parameters_file(this->parameter_file);
    }

    virtual void
    set_boundary_conditions() = 0;

    virtual void
    set_field_conditions() = 0;

    virtual void
    create_spatial_discretization() = 0;

    virtual void
    create()
    {
      create_spatial_discretization();
      AssertThrow(
        this->triangulation,
        ExcMessage(
          "It seems that your SimulationBase object does not contain"
          " a valid triangulation object. A shared_ptr to your triangulation"
          " must be specified as follows for a serialized triangulation "
          " this->triangulation = std::make_shared<Triangulation<dim>>(); "
          " or for a parallel triangulation "
          " this->triangulation = std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator); "));
      set_boundary_conditions();
      set_field_conditions();
    // @todo: shift to problems
    //   AssertThrow(
    //     this->field_conditions.initial_field,
    //     ExcMessage(
    //       "It seems that your SimulationBase object does not contain "
    //       "a valid initial field function. A shared_ptr to your initial field "
    //       "function, e.g., MyInitializeFunc<dim> must be specified as follows: "
    //       "this->field_conditions.initial_field = std::make_shared<MyInitializeFunc<dim>>();"));
    }

    /*
     * getter functions
     */
    virtual MPI_Comm
    get_mpi_communicator() const
    {
      return this->mpi_communicator;
    }
    std::shared_ptr<FieldConditions<dim>>
    get_field_conditions() const
    {
      return std::make_shared<FieldConditions<dim>>(this->field_conditions);
    }
    std::shared_ptr<TensorFunction<1, dim>>
    get_advection_field() const
    {
      return this->field_conditions.advection_field;
    }

    const BoundaryConditions<dim> &
    get_boundary_conditions() const
    {
      return this->boundary_conditions;
    }
    
    auto
    get_dirichlet_bc(const std::string problem_name) 
    {
      return boundary_conditions_map[problem_name]->dirichlet_bc;
    }
    
    auto
    get_neumann_bc(const std::string problem_name) 
    {
      return boundary_conditions_map[problem_name]->neumann_bc;
    }
  
    const
    std::vector<types::boundary_id> &
    get_no_slip_id(const std::string problem_name)
    {
      return boundary_conditions_map[problem_name]->no_slip_bc;
    }
  
    const
    std::vector<types::boundary_id> &
    get_fix_pressure_constant_id(const std::string problem_name)
    {
      return boundary_conditions_map[problem_name]->fix_pressure_constant;
    }
    
    const
    std::vector<types::boundary_id> &
    get_symmetry_id(const std::string problem_name)
    {
      return boundary_conditions_map[problem_name]->symmetry_bc;
    }
    
    auto
    get_initial_condition(const std::string operation_name) 
    {
      return field_conditions_map[operation_name]->initial_field;
    }

    template <typename FunctionType>
    void
    attach_initial_condition( std::shared_ptr<FunctionType> initial_function,
                              const std::string operation_name)
    {
      if( !field_conditions_map[operation_name] )
        field_conditions_map[operation_name] = std::make_shared<FieldConditions<dim>>();
      
      field_conditions_map[operation_name]->initial_field = initial_function;
    }
    
    template <typename FunctionType>
    void
    attach_dirichlet_boundary_condition(types::boundary_id id, 
                                        std::shared_ptr<FunctionType> boundary_function,
                                        const std::string operation_name)
    {
      if( !boundary_conditions_map[operation_name] )
        boundary_conditions_map[operation_name] = std::make_shared<BoundaryConditions<dim>>();
      
      if (boundary_conditions_map[operation_name]->dirichlet_bc.count(id) > 0)
        AssertThrow(false, ExcMessage("You try to attach a dirichlet boundary conditions "
                                      "for a boundary_id for which a boundary condition is already "
                                      "specified. Check your input related to bc!"));

      boundary_conditions_map[operation_name]->dirichlet_bc[id] = boundary_function;
    }
    
    template <typename FunctionType>
    void
    attach_neumann_boundary_condition(types::boundary_id id, 
                                      std::shared_ptr<FunctionType> boundary_function,
                                      const std::string operation_name)
    {
      if( !boundary_conditions_map[operation_name] )
        boundary_conditions_map[operation_name] = std::make_shared<BoundaryConditions<dim>>();
      
      if (boundary_conditions_map[operation_name]->neumann_bc.count(id) > 0)
        AssertThrow(false, ExcMessage("You try to attach a neumann boundary conditions "
                                      "for a boundary_id for which a boundary condition is already "
                                      "specified. Check your input related to bc!"));
      boundary_conditions_map[operation_name]->neumann_bc[id] = boundary_function;
    }
    
    void
    attach_no_slip_boundary_condition(types::boundary_id id, 
                                      const std::string operation_name)
    {
      if( !boundary_conditions_map[operation_name] )
        boundary_conditions_map[operation_name] = std::make_shared<BoundaryConditions<dim>>();
      
      auto bc = boundary_conditions_map[operation_name]->no_slip_bc;
      if ( std::find(bc.begin(), bc.end(), id)!=bc.end() )
        AssertThrow(false, ExcMessage("You try to attach a no slip boundary conditions "
                                      "for a boundary_id for which a boundary condition is already "
                                      "specified. Check your input related to bc!"));
      bc.push_back(id); 
    }
    
    void
    attach_fix_pressure_constant_condition(types::boundary_id id,        
                                      const std::string operation_name)
    {
      if( !boundary_conditions_map[operation_name] )
        boundary_conditions_map[operation_name] = std::make_shared<BoundaryConditions<dim>>();
      
     auto bc = boundary_conditions_map[operation_name]->fix_pressure_constant;
      if ( std::find(bc.begin(), bc.end(), id)!=bc.end() )
        AssertThrow(false, ExcMessage("You try to attach a no slip boundary conditions "
                                      "for a boundary_id for which a boundary condition is already "
                                      "specified. Check your input related to bc!"));
      bc.push_back(id); 
    }
    
    void
    attach_symmetry_boundary_condition(types::boundary_id id, 
                                      const std::string operation_name)
    {
      if( !boundary_conditions_map[operation_name] )
        boundary_conditions_map[operation_name] = std::make_shared<BoundaryConditions<dim>>();
      
      auto bc = boundary_conditions_map[operation_name]->symmetry_bc;
      if ( std::find(bc.begin(), bc.end(), id)!=bc.end() )
        AssertThrow(false, ExcMessage("You try to attach a no slip boundary conditions "
                                      "for a boundary_id for which a boundary condition is already "
                                      "specified. Check your input related to bc!"));
      bc.push_back(id); 
    }

    const std::string                                               parameter_file;
    const MPI_Comm                                                  mpi_communicator;
    const dealii::ConditionalOStream                                pcout;
    Parameters<double>                                              parameters;
    std::shared_ptr<Triangulation<dim, spacedim>>                   triangulation;
    FieldConditions<dim>                                            field_conditions; //@todo delete
    BoundaryConditions<dim>                                         boundary_conditions; // @todo delete
    std::map<std::string, std::shared_ptr<BoundaryConditions<dim>>> boundary_conditions_map;
    std::map<std::string, std::shared_ptr<FieldConditions<dim>>>    field_conditions_map;

  };
} // namespace MeltPoolDG
