#pragma once

#include <memory>

namespace LevelSetParallel
{
    enum BoundaryTypesLevelSet
    {
        dirichlet_bc,
        neumann_bc,
        wall_bc,
        outflow, /* do nothing */
        undefined   
    }

    template<int dim>
    struct BoundaryConditionsLevelSet
    {
        unsigned int total_number_of_bc = 0;
        // specify 
        std::map<types::boundary_id, std::shared_ptr<Function<dim>>         dirichlet_bc;
        std::map<types::boundary_id, std::shared_ptr<TensorFunction<1,dim>> neumann_bc;
        std::map<types::boundary_id, std::shared_ptr<Function<dim>>>        wall_bc;
        std::map<types::boundary_id, std::shared_ptr<Function<dim>>        outflow;
    
        inline DEAL_II_ALWAYS_INLINE
        BoundaryTypesLevelSet get_boundary_type(types::boundary_id id)
        {
            if (dirichlet_bc.find(id) != dirichlet_bc.end())   
                return BoundaryTypesLevelSet::dirichlet_bc;
            else if(neumann_bc.find(id) != neumann_bc.end())   
                return BoundaryTypesLevelSet::neumann_bc;
            else if(wall_bc.find(id) != wall_bc.end())   
                return BoundaryTypesLevelSet::wall_bc;
            /*
             * For pure convective problems no boundary conditions must be set on
             * outflow boundaries.
             */
            else if(outflow.find(id) != outflow.end())   
                return BoundaryTypesLevelSet::outflow;
            else 
                AssertThrow(false, ExcMessage("for specified boundary_id: " + std::string(id)));
                return undefined;
        }

        void verify_boundaries(const unsigned int & total_number_of_bc )
        {
        
            AssertThrow(false, ExcMessage("The number of assigned boundary conditions does not match 
                                           the total number of boundary dofs" );
        
        }
    };
}
