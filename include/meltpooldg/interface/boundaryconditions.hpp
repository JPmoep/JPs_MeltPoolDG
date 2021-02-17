/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, UIBK/TUM, October 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <map>
#include <memory>

namespace MeltPoolDG
{
  using namespace dealii;

  enum class BoundaryTypes
  {
    dirichlet_bc,
    neumann_bc,
    outflow,
    no_slip_bc,
    fix_pressure_constant,
    symmetry_bc,
    undefined
  };

  template <int dim>
  struct BoundaryConditions
  {
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> dirichlet_bc;
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> neumann_bc;
    std::vector<types::boundary_id>                              outflow;
    std::vector<types::boundary_id>                              no_slip_bc;
    std::vector<types::boundary_id>                              fix_pressure_constant;
    std::vector<types::boundary_id>                              symmetry_bc;
    std::vector<types::boundary_id>                              open_boundary_bc;

    inline DEAL_II_ALWAYS_INLINE BoundaryTypes
    get_boundary_type(types::boundary_id id)
    {
      if (dirichlet_bc.find(id) != dirichlet_bc.end())
        return BoundaryTypes::dirichlet_bc;
      else if (neumann_bc.find(id) != neumann_bc.end())
        return BoundaryTypes::neumann_bc;
      else if (std::find(outflow.begin(), outflow.end(), id) != outflow.end())
        return BoundaryTypes::outflow;
      else if (std::find(no_slip_bc.begin(), no_slip_bc.end(), id) != no_slip_bc.end())
        return BoundaryTypes::no_slip_bc;
      else if (std::find(fix_pressure_constant.begin(), fix_pressure_constant.end(), id) !=
               fix_pressure_constant.end())
        return BoundaryTypes::fix_pressure_constant;
      else if (std::find(symmetry_bc.begin(), symmetry_bc.end(), id) != symmetry_bc.end())
        return BoundaryTypes::symmetry_bc;
      else
        {
          AssertThrow(false, ExcMessage("for specified boundary_id: " + std::string(id)));
          return BoundaryTypes::undefined;
        }
    }

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
