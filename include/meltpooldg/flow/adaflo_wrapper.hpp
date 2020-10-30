#pragma once

#ifdef MELT_POOL_DG_WITH_ADAFLO

#  include <adaflo/navier_stokes.h>
#  include <adaflo/parameters.h>

#include <meltpooldg/flow/adaflo_wrapper_parameters.hpp>
#include <meltpooldg/flow/flow_base.hpp>
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG
{
namespace Flow
{
    template <int dim>
    class AdafloWrapper : public FlowBase
    {
    public:

      /**
       * Constructor.
       */
      template<int space_dim, typename number, typename VectorizedArrayType>
      AdafloWrapper(ScratchData<dim, space_dim, number, VectorizedArrayType> & scratch_data, 
                    const unsigned int idx,
                    std::shared_ptr<SimulationBase<dim>> base_in ) 
                    : dof_handler_meltpool(scratch_data.get_dof_handler(idx)) 
                    , navier_stokes(
                       base_in->parameters.adaflo_params.get_parameters(),
                        *const_cast<parallel::distributed::Triangulation<dim> *>(dynamic_cast<const parallel::distributed::Triangulation<dim> *>(&scratch_data.get_triangulation()))
                      )
      {
        /*
         * Boundary conditions for the velocity field
         */
        for (const auto& symmetry_id : base_in->get_symmetry_id("navier_stokes_u"))
          navier_stokes.set_symmetry_boundary(symmetry_id);
        for (const auto& no_slip_id : base_in->get_no_slip_id("navier_stokes_u"))
          navier_stokes.set_no_slip_boundary(no_slip_id);
        for (const auto& fix_pressure_constant_id : base_in->get_fix_pressure_constant_id("navier_stokes"))
          navier_stokes.fix_pressure_constant(fix_pressure_constant_id);
        for (const auto& dirichlet_bc : base_in->get_dirichlet_bc("navier_stokes_u"))
          navier_stokes.set_velocity_dirichlet_boundary(dirichlet_bc.first, dirichlet_bc.second);
        /*
         * Boundary conditions for the pressure field
         */
        for (const auto& neumann_bc : base_in->get_neumann_bc("navier_stokes_p"))
          navier_stokes.set_open_boundary_with_normal_flux(neumann_bc.first, neumann_bc.second);
        // @ todo: is this correct?
        for (const auto& dirichlet_bc : base_in->get_dirichlet_bc("navier_stokes_p"))
          navier_stokes.fix_pressure_constant(dirichlet_bc.first, dirichlet_bc.second);
        /*
         * Initial conditions of the navier stokes problem
         */
       AssertThrow(base_in->get_initial_condition("navier_stokes_u"),
         ExcMessage(
           "It seems that your SimulationBase object does not contain "
           "a valid initial field function. A shared_ptr to your initial field "
           "function, e.g., MyInitializeFunc<dim> must be specified as follows: "
           "this->field_conditions.initial_field = std::make_shared<MyInitializeFunc<dim>>();"));
        navier_stokes.setup_problem(*base_in->get_initial_condition("navier_stokes_u"));

      }

      /**
       * Solver time step
       */
      void
      solve() override
      {          
        navier_stokes.advance_time_step();
      }

      void
      get_velocity(LinearAlgebra::distributed::BlockVector<double> & vec) const override
      {
        VectorTools::convert_fe_sytem_vector_to_block_vector(navier_stokes.solution.block(0), 
                navier_stokes.get_dof_handler_u(), vec, dof_handler_meltpool);
      }

      void
      set_surface_tension(const LinearAlgebra::distributed::BlockVector<double> & vec) override
      {
        VectorTools::convert_block_vector_to_fe_sytem_vector(vec, 
          dof_handler_meltpool, navier_stokes.user_rhs.block(0), navier_stokes.get_dof_handler_u());
      }

      virtual 
      VectorizedArray<double> &
      get_density(const unsigned int cell, const unsigned int q)
      {
        return navier_stokes.get_matrix().begin_densities(cell)[q];   
      }
      
      virtual 
      const VectorizedArray<double> &
      get_density(const unsigned int cell, const unsigned int q) const
      {
        return navier_stokes.get_matrix().begin_densities(cell)[q];   
      }
      
      virtual 
      VectorizedArray<double> &
      get_viscosity(const unsigned int cell, const unsigned int q)
      {
        return navier_stokes.get_matrix().begin_viscosities(cell)[q];   
      }
      
      virtual 
      const VectorizedArray<double> &
      get_viscosity(const unsigned int cell, const unsigned int q) const
      {
        return navier_stokes.get_matrix().begin_viscosities(cell)[q];   
      }

    private:
      /**
       * 
       */
        const DoFHandler<dim> & dof_handler_meltpool;
        
      /**
       * Reference to the actual Navier-Stokes solver from adaflo
       */
      NavierStokes<dim> navier_stokes;
    };

    /**
     * Dummy specialization for 1. Needed to be able to compile
     * since the adaflo Navier-Stokes solver is not compiled for
     * 1D - due to the dependcy to parallel::distributed::Triangulation
     * and p4est.
     */
    template <>
    class AdafloWrapper<1> : public FlowBase
    {
    public:
      /**
       * Dummy constructor.
       */
      template<int space_dim, typename number, typename VectorizedArrayType>
      AdafloWrapper(ScratchData<1, space_dim, number, VectorizedArrayType> & scratch_data, const unsigned int idx,
                    std::shared_ptr<SimulationBase<1>> base_in)
      {
        (void) scratch_data;
        (void) idx;
        (void) base_in;

        AssertThrow(false, ExcNotImplemented ());
      }


      void
      get_velocity(LinearAlgebra::distributed::BlockVector<double> &) const override
      {
        AssertThrow(false, ExcNotImplemented ());
      }

      void
      set_surface_tension(const LinearAlgebra::distributed::BlockVector<double> & ) override
      {
        AssertThrow(false, ExcNotImplemented ());
      }

      void
      solve() override
      {
        AssertThrow(false, ExcNotImplemented ());
      }
      
      virtual 
      VectorizedArray<double> &
      get_density(const unsigned int cell, const unsigned int q)
      {
        AssertThrow(false, ExcNotImplemented ());
        (void) cell;
        (void) q;
        return dummy;   
      }
      
      virtual 
      const VectorizedArray<double> &
      get_density(const unsigned int cell, const unsigned int q) const
      {
        AssertThrow(false, ExcNotImplemented ());
        (void) cell;
        (void) q;
        return dummy;   
      }
      
      virtual 
      VectorizedArray<double> &
      get_viscosity(const unsigned int cell, const unsigned int q)
      {
        AssertThrow(false, ExcNotImplemented ());
        (void) cell;
        (void) q;
        return dummy;   
      }
      
      virtual 
      const VectorizedArray<double> &
      get_viscosity(const unsigned int cell, const unsigned int q) const
      {
        AssertThrow(false, ExcNotImplemented ());
        (void) cell;
        (void) q;
        return dummy;   
      }
      
      private:
          VectorizedArray<double> dummy;
    };

} // namespace Flow
} // namespace MeltPoolDG

#endif
