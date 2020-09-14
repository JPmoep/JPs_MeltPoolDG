#pragma once
// MeltPoolDG
#include <meltpooldg/interface/problembase.hpp>
#include <meltpooldg/advection_diffusion/advection_diffusion_problem.hpp>
#include <meltpooldg/level_set/levelset.hpp>
#include <meltpooldg/level_set_refactored/level_set_problem.hpp>
#include <meltpooldg/reinitialization/reinitialization.hpp>
#include <meltpooldg/reinitialization_refactored/reinitialization_problem.hpp>
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
        //if( sim->parameters.problem_name == "level_set_deprecated" )
          //return std::make_shared<LevelSetEquation<dim,degree>>(sim); 

        if( sim->parameters.problem_name == "level_set" )
          return std::make_shared<LevelSet::LevelSetProblem<dim,degree>>(sim); 

        //else if( sim->parameters.problem_name == "reinitialization_deprecated" )
          //return std::make_shared<Reinitialization<dim,degree>>(sim);     

        else if( sim->parameters.problem_name == "reinitialization" )
          return std::make_shared<ReinitializationNew::ReinitializationProblem<dim,degree>>();     
        
        else if( sim->parameters.problem_name == "advection_diffusion" )
          return std::make_shared<AdvectionDiffusion::AdvectionDiffusionProblem<dim,degree>>(sim);     
        
        /* add your problem here*/
        
        else
          AssertThrow(false, ExcMessage("The solver for your requested problem type does not exist"));
      }  
  };
} // namespace MeltPoolDG
