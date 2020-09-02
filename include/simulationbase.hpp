#pragma once

// dealii
#include <deal.II/distributed/tria.h>
// multiphaseflow
#include "levelsetparameters.hpp"
#include "levelsetParallel.hpp"
#include "boundaryconditions.hpp"
#include "fieldconditions.hpp"
// c++
#include <memory>
namespace LevelSetParallel
{
    using namespace dealii;

    template <int dim>
    class SimulationBase
    {
      public:
        
        //virtual void set_mpi_commun() = 0;
        SimulationBase(MPI_Comm my_communicator)
        : mpi_communicator(my_communicator)
        , triangulation(my_communicator)
        {
        }

        virtual ~SimulationBase()
        {}

        virtual void set_parameters() = 0;
        
        virtual void set_boundary_conditions() = 0;
        
        virtual void set_field_conditions() = 0;

        virtual void create_spatial_discretization() = 0;
        
        virtual MPI_Comm get_mpi_communicator() const { return this->mpi_communicator; };
        
        std::shared_ptr<FieldConditions<dim>>     get_field_conditions() const { return std::make_shared<FieldConditions<dim>>(this->field_conditions); }
        
        std::shared_ptr<BoundaryConditions<dim>>  get_boundary_conditions() const { return std::make_shared<BoundaryConditions<dim>>(this->boundary_conditions); }

        const MPI_Comm                            mpi_communicator;
        parallel::distributed::Triangulation<dim> triangulation; 
        FieldConditions<dim>                      field_conditions;
        BoundaryConditions<dim>                   boundary_conditions;

        LevelSetParameters                        parameters;

    };
}
