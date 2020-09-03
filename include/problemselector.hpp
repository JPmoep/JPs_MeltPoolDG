#pragma once
// multiphaseflow
#include "levelsetParallel.hpp"
#include "simulationbase.hpp"
#include "problembase.hpp"

//@todo: merge this file with problembase.hpp

namespace LevelSetParallel
{
  using namespace dealii;

  template<int dim, int degree>
  class ProblemSelector
  {
    private:
      typedef ProblemBase<dim,degree> ProblemType;
    public:
      static 
      std::shared_ptr<ProblemBase<dim,degree>> 
      get_problem( std::shared_ptr<SimulationBase<dim>> sim )
      {
        std::shared_ptr<ProblemType> problem;
        if( sim->parameters.problem_name == "levelset" )
          return std::make_shared<LevelSetEquation<dim,degree>>(sim);     
        if( sim->parameters.problem_name == "reinitialization" )
          return std::make_shared<Reinitialization<dim,degree>>(sim);     
        else
          AssertThrow(false, ExcMessage("The solver for your requested problem solver does not exist"));
      }  
  };
} // namespace LevelSetParallel
