#pragma once
// MeltPoolDG
#include <meltpooldg/levelset/levelset.hpp>
#include <meltpooldg/reinitialization/reinitialization.hpp>
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/interface/problembase.hpp>

//@todo: merge this file with problembase.hpp

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
        if( sim->parameters.problem_name == "levelset" )
          return std::make_shared<LevelSetEquation<dim,degree>>(sim);     
        if( sim->parameters.problem_name == "reinitialization" )
          return std::make_shared<Reinitialization<dim,degree>>(sim);     
        else
          AssertThrow(false, ExcMessage("The solver for your requested problem type does not exist"));
      }  
  };
} // namespace MeltPoolDG
