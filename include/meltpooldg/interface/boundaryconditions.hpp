#pragma once
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <map>
#include <memory>

namespace MeltPoolDG
{
  using namespace dealii;

  template <int dim>
  struct BoundaryConditions
  {
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>>  dirichlet_bc;
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>>  neumann_bc;
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>>  outflow;
    std::vector<types::boundary_id>                               no_slip_bc;
    std::vector<types::boundary_id>                               fix_pressure_constant;
    std::vector<types::boundary_id>                               symmetry_bc;

    //inline DEAL_II_ALWAYS_INLINE BoundaryTypesLevelSet
    //get_boundary_type(types::boundary_id id)
    //{
      //if (dirichlet_bc.find(id) != dirichlet_bc.end())
        //return BoundaryTypesLevelSet::dirichlet_bc;
      //// else if(neumann_bc.find(id) != neumann_bc.end())
      //// return BoundaryTypesLevelSet::neumann_bc;
      //// else if(wall_bc.find(id) != wall_bc.end())
      //// return BoundaryTypesLevelSet::wall_bc;
      /*
       * For pure convective problems no boundary conditions must be set on
       * outflow boundaries.
       */
      //else if (outflow.find(id) != outflow.end())
        //return BoundaryTypesLevelSet::outflow;
      //else
        //{
          //AssertThrow(false, ExcMessage("for specified boundary_id: " + std::string(id)));

          //return BoundaryTypesLevelSet::undefined;
        //}
    //}
    //

    void
    verify_boundaries(const unsigned int &total_number_of_bc)
    {
      // @todo: add a procedure to verify if boundary conditions are set correctly
      AssertThrow(
        false,
        ExcMessage(
          "The number of assigned boundary conditions does not match the total number of boundary dofs"));
    }
  };
} // namespace MeltPoolDG
