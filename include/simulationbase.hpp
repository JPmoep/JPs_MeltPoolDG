#pragma once

#include "levelsetparameters.hpp"
#include "levelsetParallel.hpp"
#include "boundaryconditions.hpp"
#include "fieldconditions.hpp"
#include <memory>

namespace LevelSetParallel
{
    using namespace dealii;

    template <int dim>
    class SimulationBase
    {
        public:
    
            virtual void set_parameters() = 0;
            
            virtual void set_boundary_conditions() = 0;
            
            virtual void set_field_conditions() = 0;

            virtual void create_spatial_discretization() = 0;

            LevelSetParameters parameters;
            FieldConditions<dim>               field_conditions;
            BoundaryConditionsLevelSet<dim>    boundary_conditions;
    };

}
