#pragma once
// MeltPoolDG
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/utilities/utilityfunctions.hpp>
// simulations
#include <meltpooldg/simulations/advection_diffusion/advection_diffusion.hpp>
#include <meltpooldg/simulations/flow_past_cylinder/flow_past_cylinder.hpp>
#include <meltpooldg/simulations/recoil_pressure/recoil_pressure.hpp>
#include <meltpooldg/simulations/reinit_circle/reinit_circle.hpp>
#include <meltpooldg/simulations/rising_bubble/rising_bubble.hpp>
#include <meltpooldg/simulations/rotating_bubble/rotating_bubble.hpp>
#include <meltpooldg/simulations/slotted_disc/slotted_disc.hpp>
#include <meltpooldg/simulations/spurious_currents/spurious_currents.hpp>
#include <meltpooldg/simulations/vortex_bubble/vortex_bubble.hpp>

namespace MeltPoolDG
{
  namespace Simulation
  {
    template <int dim>
    class SimulationSelector
    {
    public:
      static std::shared_ptr<SimulationBase<dim>>
      get_simulation(const std::string simulation_name,
                     const std::string parameter_file,
                     const MPI_Comm    mpi_communicator)
      {
        if (simulation_name == "reinit_circle")
          return std::make_shared<ReinitCircle::SimulationReinit<dim>>(parameter_file,
                                                                       mpi_communicator);
        else if (simulation_name == "advection_diffusion")
          return std::make_shared<AdvectionDiffusion::SimulationAdvec<dim>>(parameter_file,
                                                                            mpi_communicator);
        else if (simulation_name == "rotating_bubble")
          return std::make_shared<RotatingBubble::SimulationRotatingBubble<dim>>(parameter_file,
                                                                                 mpi_communicator);
        else if (simulation_name == "flow_past_cylinder")
          return std::make_shared<FlowPastCylinder::SimulationFlowPastCylinder<dim>>(
            parameter_file, mpi_communicator);
        else if (simulation_name == "spurious_currents")
          return std::make_shared<SpuriousCurrents::SimulationSpuriousCurrents<dim>>(
            parameter_file, mpi_communicator);
        else if (simulation_name == "rising_bubble")
          return std::make_shared<RisingBubble::SimulationRisingBubble<dim>>(parameter_file,
                                                                             mpi_communicator);
        else if (simulation_name == "slotted_disc")
          return std::make_shared<SlottedDisc::SimulationSlottedDisc<dim>>(parameter_file,
                                                                           mpi_communicator);
        else if (simulation_name == "recoil_pressure")
          return std::make_shared<RecoilPressure::SimulationRecoilPressure<dim>>(parameter_file,
                                                                                 mpi_communicator);
        else if (simulation_name == "vortex_bubble")
          return std::make_shared<VortexBubble::SimulationVortexBubble<dim>>(parameter_file,
                                                                             mpi_communicator);
        /* add your simulation here*/
        else
          AssertThrow(false,
                      ExcMessage("The input-file for your requested application does not exist"));
      }
    };
  } // namespace Simulation
} // namespace MeltPoolDG
