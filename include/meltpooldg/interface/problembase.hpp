#pragma once
#include <meltpooldg/interface/simulationbase.hpp>

namespace MeltPoolDG
{
  using namespace dealii;

  template <int dim>
  class ProblemBase
  {
    public:
      virtual ~ProblemBase()
      {}

      virtual void run( std::shared_ptr<SimulationBase<dim>> base_in ) = 0;
      virtual std::string get_name() = 0;
      virtual void perform_convergence_study() {};
  };

} // namespace MeltPoolDG
