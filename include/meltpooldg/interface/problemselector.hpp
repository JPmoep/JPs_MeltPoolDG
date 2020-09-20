#pragma once
// MeltPoolDG
#include <meltpooldg/interface/problembase.hpp>
#include <meltpooldg/advection_diffusion/advection_diffusion_problem.hpp>
#include <meltpooldg/level_set/level_set_problem.hpp>
#include <meltpooldg/reinitialization/reinitialization_problem.hpp>
#include <meltpooldg/interface/simulationbase.hpp>
/* add your problem here*/

namespace MeltPoolDG
{
  using namespace dealii;

  template<int dim, int degree>
  class ProblemSelector
  {
    public:
      static 
      std::shared_ptr<ProblemBase<dim,degree>> 
      get_problem( std::shared_ptr<SimulationBase<dim>> sim )
      {
        std::shared_ptr<ProblemBase<dim,degree>> problem;
        if( sim->parameters.problem_name == "level_set" )
          return std::make_shared<LevelSet::LevelSetProblem<dim,degree>>(); 

        else if( sim->parameters.problem_name == "reinitialization" )
          return std::make_shared<Reinitialization::ReinitializationProblem<dim,degree>>();     
        
        else if( sim->parameters.problem_name == "advection_diffusion" )
          return std::make_shared<AdvectionDiffusion::AdvectionDiffusionProblem<dim,degree>>();     
        
        /* add your problem here*/
        
        else
          AssertThrow(false, ExcMessage("The solver for your requested problem type does not exist"));
      }  
  };
} // namespace MeltPoolDG
