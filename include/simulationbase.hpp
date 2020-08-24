#pragma once

#include "levelsetparameters.hpp"
#include "levelsetParallel.hpp"
#include "boundaryconditions.hpp"
#include "fieldconditions.hpp"

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
        protected:
            LevelSetParameters parameters;
    };

}
