#pragma once
// MeltPoolDG
#include <meltpooldg/advection_diffusion/advection_diffusion_problem.hpp>
#include <meltpooldg/flow/two_phase_flow_problem.hpp>
#include <meltpooldg/interface/problembase.hpp>
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/level_set/level_set_problem.hpp>
#include <meltpooldg/reinitialization/reinitialization_problem.hpp>
/* add your problem here*/

namespace MeltPoolDG
{
  using namespace dealii;

  template <int dim>
  class ProblemSelector
  {
  public:
    static std::shared_ptr<ProblemBase<dim>>
    get_problem(std::string problem_name)
    {
      if (problem_name == "level_set" || problem_name == "level_set_with_evaporation")
        return std::make_shared<LevelSet::LevelSetProblem<dim>>();

      else if (problem_name == "reinitialization")
        return std::make_shared<Reinitialization::ReinitializationProblem<dim>>();

      else if (problem_name == "advection_diffusion")
        return std::make_shared<AdvectionDiffusion::AdvectionDiffusionProblem<dim>>();

      else if (problem_name == "two_phase_flow" || problem_name == "melt_pool" ||
               problem_name == "two_phase_flow_with_evaporation")
        return std::make_shared<Flow::TwoPhaseFlowProblem<dim>>();
      /* add your problem here*/

      else
        AssertThrow(false, ExcMessage("The solver for your requested problem type does not exist"));
    }
  };
} // namespace MeltPoolDG
