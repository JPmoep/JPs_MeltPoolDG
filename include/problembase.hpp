#pragma once

namespace MeltPoolDG
{
  using namespace dealii;

  template <int dim, int degree>
  class ProblemBase
  {
    public:
      virtual ~ProblemBase()
      {}

      virtual void run() = 0;
      virtual std::string get_name() = 0;
      virtual void perform_convergence_study() {};
  };

} // namespace MeltPoolDG
