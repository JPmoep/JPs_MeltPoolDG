// deal-specific libraries
#include <deal.II/base/mpi.h>
// c++
#include <iostream>
// MeltPoolDG
#include <meltpooldg/interface/problem_selector.hpp>
#include <meltpooldg/simulations/simulation_selector.hpp>
// simulations

namespace MeltPoolDG
{
  namespace Simulation
  {
    template <typename number = double>
    void
    run_simulation(const std::string parameter_file, const MPI_Comm mpi_communicator)
    {
      Parameters<number> parameters;
      parameters.process_parameters_file(parameter_file);

      const auto dim = parameters.base.dimension;

      try
        {
          if (dim == 1)
            {
              auto sim = SimulationSelector<1>::get_simulation(parameters.base.application_name,
                                                               parameter_file,
                                                               mpi_communicator);
              sim->create();
              if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
                parameters.print_parameters(std::cout);
              auto problem = ProblemSelector<1>::get_problem(parameters.base.problem_name);
              problem->run(sim);
            }
          else if (dim == 2)
            {
              auto sim = SimulationSelector<2>::get_simulation(parameters.base.application_name,
                                                               parameter_file,
                                                               mpi_communicator);
              sim->create();
              if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
                parameters.print_parameters(std::cout);
              auto problem = ProblemSelector<2>::get_problem(parameters.base.problem_name);
              problem->run(sim);
            }
          else if (dim == 3)
            {
              auto sim = SimulationSelector<3>::get_simulation(parameters.base.application_name,
                                                               parameter_file,
                                                               mpi_communicator);
              sim->create();
              if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
                parameters.print_parameters(std::cout);
              auto problem = ProblemSelector<3>::get_problem(parameters.base.problem_name);
              problem->run(sim);
            }
          else
            {
              AssertThrow(false, ExcMessage("Dimension must be 1, 2 or 3."));
            }
        }
      catch (std::exception &exc)
        {
          std::cerr << std::endl
                    << std::endl
                    << "----------------------------------------------------" << std::endl;
          std::cerr << "Exception on processing: " << std::endl
                    << exc.what() << std::endl
                    << "Aborting!" << std::endl
                    << "----------------------------------------------------" << std::endl;
        }
      catch (...)
        {
          std::cerr << std::endl
                    << std::endl
                    << "----------------------------------------------------" << std::endl;
          std::cerr << "Unknown exception!" << std::endl
                    << "Aborting!" << std::endl
                    << "----------------------------------------------------" << std::endl;
        }
    }

  } // namespace Simulation
} // namespace MeltPoolDG

int
main(int argc, char *argv[])
{
  using namespace dealii;
  using namespace MeltPoolDG;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPI_Comm mpi_comm(MPI_COMM_WORLD);


  std::string input_file;
  if (argc >= 2)
    {
      input_file = std::string(argv[argc - 1]);
      Simulation::run_simulation(input_file, mpi_comm);
    }
  else
    AssertThrow(false, ExcMessage("no input file specified"));

  return 0;
}
