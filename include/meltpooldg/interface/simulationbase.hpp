#pragma once
// dealii
#include <deal.II/distributed/tria.h>
// MeltPoolDG
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/interface/boundaryconditions.hpp>
#include <meltpooldg/interface/fieldconditions.hpp>
// c++
#include <memory>

namespace MeltPoolDG
{
    using namespace dealii;

    template <int dim, int spacedim=dim>
    class SimulationBase
    {
        public:

        SimulationBase(MPI_Comm my_communicator)
        : mpi_communicator(my_communicator)
        , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0 )
        {
        }

        virtual ~SimulationBase() = default;

        virtual void set_parameters() = 0;
        
        virtual void set_boundary_conditions() = 0;
        
        virtual void set_field_conditions() = 0;

        virtual void create_spatial_discretization() = 0;

        virtual void create()
        { 
          create_spatial_discretization();
          set_boundary_conditions();
          set_field_conditions();
        };

        /*
         * getter functions
        */

        virtual MPI_Comm                          get_mpi_communicator() const { return this->mpi_communicator; };
        
        std::shared_ptr<FieldConditions<dim>>     get_field_conditions() const { return std::make_shared<FieldConditions<dim>>(this->field_conditions); }
        std::shared_ptr<TensorFunction<1,dim>>    get_advection_field()  const { return this->field_conditions.advection_field; }
        
        const BoundaryConditions<dim>&            get_boundary_conditions() const { return this->boundary_conditions; }

        const MPI_Comm                                 mpi_communicator;
        const dealii::ConditionalOStream               pcout;          // @todo: make protected
        Parameters<double>                             parameters;
        std::shared_ptr<Triangulation<dim,spacedim>>   triangulation;  // @todo: make protected

      //protected:
        FieldConditions<dim>                           field_conditions;
        BoundaryConditions<dim>                        boundary_conditions;
    };
} // namespace MeltPoolDG
