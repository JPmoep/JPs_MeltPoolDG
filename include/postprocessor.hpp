#pragma once
// for distributed vectors/matrices
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/distributed/tria.h>
// for TableHandler
#include <deal.II/base/table_handler.h>
#include <deal.II/base/conditional_ostream.h> 

using namespace dealii;

namespace LevelSetParallel
{
  
  template <int dim>
  class Postprocessor
  {
    private:
      typedef LinearAlgebra::distributed::Vector<double>      VectorType;
      std::vector<std::vector<double>>                        volumes;
      TableHandler                                            volume_table;
      ConditionalOStream                         pcout;
    public:
    Postprocessor(const MPI_Comm& mpi_communicator)
      : pcout(               std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator) == 0) )
    {
    }
    
    
    void compute_error( const int                                          n_q_points,
                        const VectorType&                                  approximate_solution,
                        const Function<dim>&                               ExactSolution,
                        const DoFHandler<dim>&                             dof_handler,
                        const parallel::distributed::Triangulation<dim>&   triangulation)
    {
      
      const auto qGauss = QGauss<dim>(n_q_points);
      Vector<double> norm_per_cell(triangulation.n_active_cells());

      VectorTools::integrate_difference(dof_handler,
                                        approximate_solution,
                                        ExactSolution,
                                        norm_per_cell,
                                        qGauss,
                                        VectorTools::L2_norm);
      
      pcout     << "L2 error =    "
                << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                << compute_global_error(triangulation, 
                                        norm_per_cell,
                                        VectorTools::L2_norm) << std::endl;

      Vector<double> difference_per_cell(triangulation.n_active_cells());

      VectorTools::integrate_difference(dof_handler,
                                        approximate_solution,
                                        ExactSolution,
                                        difference_per_cell,
                                        qGauss,
                                        VectorTools::L1_norm);

      double h1_error = VectorTools::compute_global_error(triangulation,
                                                          difference_per_cell,
                                                          VectorTools::L1_norm);
      pcout << "L1 error = " << h1_error << std::endl;
    }
    
    std::vector<double> compute_volume_of_phases( const int      degree, 
                                          const int              n_q_points,
                                          const DoFHandler<dim>& dof_handler,
                                          const VectorType&      solution_levelset,
                                          const double           time,
                                          const MPI_Comm&        mpi_communicator,
                                          const double           max_value = 1,
                                          const double           min_value = -1.0
                                        )
    {
      FE_Q<dim> fe(degree);
      FEValues<dim> fe_values( fe,
                               QGauss<dim>( n_q_points ),
                               update_values | update_JxW_values | update_quadrature_points );

      std::vector<double> phi_at_q(  QGauss<dim>( n_q_points ).size() );

      std::vector<double> volume_fraction;
      double vol_phase_1 = 0;
      double vol_phase_2 = 0;
      const double threshhold = 0.5;

      for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
          fe_values.reinit(               cell );
          fe_values.get_function_values(  solution_levelset, phi_at_q ); // compute values of old solution

          for (const unsigned int q_index : fe_values.quadrature_point_indices())
          {
              const double phi_normalized = utilityFunctions::normalizeFunction ( phi_at_q[q_index], min_value, max_value );
              if (phi_normalized>=threshhold)
                  vol_phase_1 += fe_values.JxW(q_index);
              else 
                  vol_phase_2 += fe_values.JxW(q_index);
          }
      }
      volume_fraction.emplace_back(Utilities::MPI::sum(vol_phase_1, mpi_communicator));
      volume_fraction.emplace_back(Utilities::MPI::sum(vol_phase_2, mpi_communicator));
      //@ todo: write template class for formatted output table
      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) 
          std::cout << "vol phase 1: " << volume_fraction[0] << " vol phase 2: " << volume_fraction[1] << std::endl;
        
      volume_table.add_value("time", time);
      volume_table.add_value("vol phase 1", volume_fraction[0]);
      volume_table.add_value("vol phase 2", volume_fraction[1]);

      return volume_fraction;
    }

    void collect_volume_fraction(const std::vector<double>& volume_fraction)
    {
      volumes.emplace_back(volume_fraction);
    }

    void print_volume_fraction_table(const MPI_Comm& mpi_communicator)
    {
      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) 
        volume_table.write_text(std::cout);
    }


    
    //size_t headerWidths[2] = {
          //std::string("time").size(),
          //std::string("volume phase 1").size()
      //};

      //if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      //{
          //if ( time==parameters.start_time )
          //{
              //std::cout << "output file opened" << std::endl;
              //std::fstream fs;
              //fs.open (parameters.filename_volume_output, std::fstream::out);
              //fs.precision(10);
              //fs << "time | volume phase 1 | volume phase 2 " << std::endl; 
              //fs << std::left << std::setw(headerWidths[0]) << time;
              //fs << "   " << std::left << std::setw(headerWidths[1]) << volume_fraction[0]; 
              //fs << "   " << std::left << std::setw(headerWidths[1]) << volume_fraction[1] << std::endl; 
              //fs.close();
          //}
          //else
          //{
              //std::fstream fs;
              //fs.open (parameters.filename_volume_output,std::fstream::in | std::fstream::out | std::fstream::app);
              //fs.precision(10);
              //fs << std::left << std::setw(headerWidths[0]) << time;
              //fs << "   " << std::left << std::setw(headerWidths[1]) << volume_fraction[0]; 
              //fs << "   " << std::left << std::setw(headerWidths[1]) << volume_fraction[1] << std::endl; 
          //}
      
  };

}
