#pragma once
// MeltPoolDG
#include <meltpooldg/utilities/utilityfunctions.hpp>
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/interface/problem_selector.hpp>
// simulations
#include <meltpooldg/simulations/reinit_circle/reinit_circle.hpp>
#include <meltpooldg/simulations/reinit_circle_amr/reinit_circle_amr.hpp>
#include <meltpooldg/simulations/advection_diffusion/advection_diffusion.hpp>
#include <meltpooldg/simulations/rotating_bubble/rotating_bubble.hpp>

namespace MeltPoolDG
{
namespace Simulation
{
  template<int dim>
  class SimulationSelector
  {
    public:
      static 
      std::shared_ptr<SimulationBase<dim>> 
      get_simulation( const std::string simulation_name,
                      const std::string parameter_file,
                      const MPI_Comm mpi_communicator )
      {
        if( simulation_name == "reinit_circle" )
          return std::make_shared<ReinitCircle::SimulationReinit<dim>>(parameter_file,
                                                                       mpi_communicator);
        else if( simulation_name == "reinit_circle_amr" )
        {
          return std::make_shared<ReinitCircleAMR::SimulationReinit<dim>>(parameter_file,
                                                                            mpi_communicator);
        }
        else if( simulation_name == "advection_diffusion" )
        {
          return std::make_shared<AdvectionDiffusion::SimulationAdvec<dim>>(parameter_file,
                                                                            mpi_communicator);
        }
        else if( simulation_name == "rotating_bubble" )
        {
          return std::make_shared<RotatingBubble::SimulationRotatingBubble<dim>>(parameter_file,
                                                                                 mpi_communicator);
        }
          /* add your simulation here*/
        else
          AssertThrow(false, ExcMessage("The input-fle for your requested application does not exist"));
      }  
  };
} // namespace Simulation
} // namespace MeltPoolDG