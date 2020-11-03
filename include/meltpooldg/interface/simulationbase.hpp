#pragma once
// dealii
#include <deal.II/base/exceptions.h>

#include <deal.II/distributed/tria.h>
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
    }

    /**
     * Attach functions for field conditions
     */
    template <typename FunctionType>
    void
    attach_initial_condition(std::shared_ptr<FunctionType> initial_function,
                             const std::string             operation_name)
    {
      if (!field_conditions_map[operation_name])
        field_conditions_map[operation_name] = std::make_shared<FieldConditions<dim>>();

      field_conditions_map[operation_name]->initial_field = initial_function;
    }

    template <typename FunctionType>
    void
    attach_advection_field(std::shared_ptr<FunctionType> advection_velocity,
                           const std::string             operation_name)
    {
      if (!field_conditions_map[operation_name])
        field_conditions_map[operation_name] = std::make_shared<FieldConditions<dim>>();

      field_conditions_map[operation_name]->advection_field = advection_velocity;
    }

    template <typename FunctionType>
    void
    attach_exact_solution(std::shared_ptr<FunctionType> exact_solution,
                          const std::string             operation_name)
    {
      if (!field_conditions_map[operation_name])
        field_conditions_map[operation_name] = std::make_shared<FieldConditions<dim>>();

      field_conditions_map[operation_name]->exact_solution_field = exact_solution;
    }

    /**
     * Attach functions for boundary conditions
     */

    template <typename FunctionType>
    void
    attach_dirichlet_boundary_condition(types::boundary_id            id,
                                        std::shared_ptr<FunctionType> boundary_function,
                                        const std::string             operation_name)
    {
      if (!boundary_conditions_map[operation_name])
        boundary_conditions_map[operation_name] = std::make_shared<BoundaryConditions<dim>>();

      if (boundary_conditions_map[operation_name]->dirichlet_bc.count(id) > 0)
        AssertThrow(false,
                    ExcMessage("You try to attach a dirichlet boundary conditions "
                               "for a boundary_id for which a boundary condition is already "
                               "specified. Check your input related to bc!"));

      boundary_conditions_map[operation_name]->dirichlet_bc[id] = boundary_function;
    }

    template <typename FunctionType>
    void
    attach_neumann_boundary_condition(types::boundary_id            id,
                                      std::shared_ptr<FunctionType> boundary_function,
                                      const std::string             operation_name)
    {
      if (!boundary_conditions_map[operation_name])
        boundary_conditions_map[operation_name] = std::make_shared<BoundaryConditions<dim>>();

      if (boundary_conditions_map[operation_name]->neumann_bc.count(id) > 0)
        AssertThrow(false,
                    ExcMessage("You try to attach a neumann boundary conditions "
                               "for a boundary_id for which a boundary condition is already "
                               "specified. Check your input related to bc!"));
      boundary_conditions_map[operation_name]->neumann_bc[id] = boundary_function;
    }

    void
    attach_no_slip_boundary_condition(types::boundary_id id, const std::string operation_name)
    {
      if (!boundary_conditions_map[operation_name])
        boundary_conditions_map[operation_name] = std::make_shared<BoundaryConditions<dim>>();

      auto &bc = boundary_conditions_map[operation_name]->no_slip_bc;
      if (std::find(bc.begin(), bc.end(), id) != bc.end())
        AssertThrow(false,
                    ExcMessage("You try to attach a no slip boundary conditions "
                               "for a boundary_id for which a boundary condition is already "
                               "specified. Check your input related to bc!"));
      bc.push_back(id);
    }

    void
    attach_fix_pressure_constant_condition(types::boundary_id id, const std::string operation_name)
    {
      if (!boundary_conditions_map[operation_name])
        boundary_conditions_map[operation_name] = std::make_shared<BoundaryConditions<dim>>();

      auto &bc = boundary_conditions_map[operation_name]->fix_pressure_constant;
      if (std::find(bc.begin(), bc.end(), id) != bc.end())
        AssertThrow(false,
                    ExcMessage("You try to attach a no slip boundary conditions "
                               "for a boundary_id for which a boundary condition is already "
                               "specified. Check your input related to bc!"));
      bc.push_back(id);
    }

    void
    attach_symmetry_boundary_condition(types::boundary_id id, const std::string operation_name)
    {
      if (!boundary_conditions_map[operation_name])
        boundary_conditions_map[operation_name] = std::make_shared<BoundaryConditions<dim>>();

      auto &bc = boundary_conditions_map[operation_name]->symmetry_bc;
      if (std::find(bc.begin(), bc.end(), id) != bc.end())
        AssertThrow(false,
                    ExcMessage("You try to attach a symmetry boundary conditions "
                               "for a boundary_id for which a boundary condition is already "
                               "specified. Check your input related to bc!"));
      bc.push_back(id);
    }

    /*
     * getter functions
     */
    virtual MPI_Comm
    get_mpi_communicator() const
    {
      return this->mpi_communicator;
    }

    /**
     * Getter functions for field conditions
     */
    const std::shared_ptr<Function<dim>> &
    get_initial_condition(const std::string operation_name)
    {
      return field_conditions_map[operation_name]->initial_field;
    }

    const std::shared_ptr<TensorFunction<1, dim>> &
    get_advection_field(const std::string operation_name)
    {
      return field_conditions_map[operation_name]->advection_field;
    }

    const std::shared_ptr<Function<dim>> &
    get_exact_solution(const std::string operation_name)
    {
      return field_conditions_map[operation_name]->exact_solution_field;
    }

    /**
     * Attach functions for boundary conditions
     */
    const auto &
    get_dirichlet_bc(const std::string operation_name)
    {
      return boundary_conditions_map[operation_name]->dirichlet_bc;
    }

    const auto &
    get_neumann_bc(const std::string operation_name)
    {
      return boundary_conditions_map[operation_name]->neumann_bc;
    }

    const std::vector<types::boundary_id> &
    get_no_slip_id(const std::string operation_name)
    {
      return boundary_conditions_map[operation_name]->no_slip_bc;
    }

    const std::vector<types::boundary_id> &
    get_fix_pressure_constant_id(const std::string operation_name)
    {
      if (!boundary_conditions_map[operation_name])
        AssertThrow(false,
                    ExcMessage(
                      "get_fix_pressure_constant_id: requested boundary condition not found"));
      return boundary_conditions_map[operation_name]->fix_pressure_constant;
    }

    const std::vector<types::boundary_id> &
    get_symmetry_id(const std::string operation_name)
    {
      if (!boundary_conditions_map[operation_name])
        AssertThrow(false, ExcMessage("get_symmetry_id: requested boundary condition not found"));
      return boundary_conditions_map[operation_name]->symmetry_bc;
    }

  public:
    const std::string                             parameter_file;
    const MPI_Comm                                mpi_communicator;
    const dealii::ConditionalOStream              pcout;
    Parameters<double>                            parameters;
    std::shared_ptr<Triangulation<dim, spacedim>> triangulation;

  private:
    std::map<std::string, std::shared_ptr<FieldConditions<dim>>>    field_conditions_map;
    std::map<std::string, std::shared_ptr<BoundaryConditions<dim>>> boundary_conditions_map;
  };
} // namespace MeltPoolDG
