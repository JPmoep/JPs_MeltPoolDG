#pragma once
// dealii
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
    SimulationBase(std::string parameter_file_in, MPI_Comm my_communicator)
    : parameter_file(parameter_file_in)
    , mpi_communicator(my_communicator)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      set_parameters();
    }

    virtual ~SimulationBase() = default;

    virtual void
    set_parameters()
    {
      this->parameters.process_parameters_file(this->parameter_file, this->pcout);
    };

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
      AssertThrow(this->triangulation,
                  ExcMessage("It seems that your SimulationBase object does not contain"
                             " a valid triangulation object. A shared_ptr to your triangulation"
                             " must be specified as follows for a serialized triangulation "
                             " this->triangulation = std::make_shared<Triangulation<dim>>(); "
                             " or for a parallel triangulation "
                             " this->triangulation = std::make_shared<parallel::distributed::Triangulation<dim>>(this->mpi_communicator); "
                            ));
      set_boundary_conditions();
      set_field_conditions();
      AssertThrow(this->field_conditions.initial_field,
                  ExcMessage("It seems that your SimulationBase object does not contain "
                             "a valid initial field function. A shared_ptr to your initial field "
                             "function, e.g., MyInitializeFunc<dim> must be specified as follows: "
                             "this->field_conditions.initial_field = std::make_shared<MyInitializeFunc<dim>>();" 
                            ));
    };

    /*
     * getter functions
     */
    virtual MPI_Comm
    get_mpi_communicator() const
    {
      return this->mpi_communicator;
    };
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

    const std::string                             parameter_file;
    const MPI_Comm                                mpi_communicator;
    const dealii::ConditionalOStream              pcout;
    Parameters<double>                            parameters;
    std::shared_ptr<Triangulation<dim, spacedim>> triangulation;
    FieldConditions<dim>                          field_conditions;
    BoundaryConditions<dim>                       boundary_conditions;
  };
} // namespace MeltPoolDG
