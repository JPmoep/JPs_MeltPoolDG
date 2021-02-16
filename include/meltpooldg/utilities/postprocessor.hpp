#pragma once
// for distributed vectors/matrices
#include <deal.II/distributed/tria.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/vector_tools.h>
// for TableHandler
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/table_handler.h>

using namespace dealii;

// @ todo: !!! clean-up and refactoring !!!

namespace MeltPoolDG
{
  template <int dim>
  class Postprocessor
  {
  private:
    using VectorType = LinearAlgebra::distributed::Vector<double>;

    std::vector<std::vector<double>> volumes;
    TableHandler                     volume_table;

    const MPI_Comm              mpi_communicator;
    const ParaviewData<double> &pv_data;
    const Mapping<dim> &        mapping;
    const Triangulation<dim> &  triangulation;
    ConditionalOStream          pcout;
    bool                        do_simplex;

    DataOut<dim> data_out;

  public:
    Postprocessor(const MPI_Comm              mpi_communicator_in,
                  const ParaviewData<double> &pv_data_in,
                  const Mapping<dim> &        mapping_in,
                  const Triangulation<dim> &  triangulation_in)
      : mpi_communicator(mpi_communicator_in)
      , pv_data(pv_data_in)
      , mapping(mapping_in)
      , triangulation(triangulation_in)
      , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
      , do_simplex(!triangulation.all_reference_cells_are_hyper_cube())
    {}

    /*
     *  This function collects and performs all relevant postprocessing steps.
     */
    void
    process(const int                                  n_time_step,
            const std::function<void(DataOut<dim> &)> &attach_output_vectors,
            const std::function<void()> &              post_operation = {})
    {
      if ((pv_data.do_output) && !(n_time_step % pv_data.write_frequency))
        {
          attach_output_vectors(data_out);

          write_paraview_files(n_time_step);

          if (pv_data.print_boundary_id)
            print_boundary_ids();
        }

      if (post_operation)
        post_operation();

      data_out.clear();
    }

  private:
    void
    write_paraview_files(const double time_step)
    {
      DataOutBase::VtkFlags flags;
      if (do_simplex == false)
        flags.write_higher_order_cells = true;
      data_out.set_flags(flags);

      data_out.build_patches(mapping);
      data_out.write_vtu_with_pvtu_record(pv_data.directory,
                                          pv_data.filename,
                                          time_step,
                                          mpi_communicator,
                                          pv_data.n_digits_timestep,
                                          pv_data.n_groups);
    }

    void
    print_boundary_ids()
    {
      const unsigned int rank    = Utilities::MPI::this_mpi_process(mpi_communicator);
      const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(mpi_communicator);

      const unsigned int n_digits = static_cast<int>(std::ceil(std::log10(std::fabs(n_ranks))));

      std::string filename =
        pv_data.filename + "_boundary_ids" + Utilities::int_to_string(rank, n_digits) + ".vtk";
      std::ofstream output(filename.c_str());

      GridOut           grid_out;
      GridOutFlags::Vtk flags;
      flags.output_cells         = false;
      flags.output_faces         = true;
      flags.output_edges         = false;
      flags.output_only_relevant = false;
      grid_out.set_flags(flags);
      grid_out.write_vtk(triangulation, output);
    }

    /*
     * @todo
     */
    void
    compute_error(const int              n_q_points,
                  const VectorType &     approximate_solution,
                  const Function<dim> &  ExactSolution,
                  const DoFHandler<dim> &dof_handler,
                  const Mapping<dim> &   mapping)
    {
      const auto &triangulation = dof_handler.get_triangulation();

      const auto     qGauss = QGauss<dim>(n_q_points);
      Vector<double> norm_per_cell(triangulation.n_active_cells());

      dealii::VectorTools::integrate_difference(mapping,
                                                dof_handler,
                                                approximate_solution,
                                                ExactSolution,
                                                norm_per_cell,
                                                qGauss,
                                                dealii::VectorTools::L2_norm);

      pcout << "L2 error =    " << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
            << compute_global_error(triangulation, norm_per_cell, dealii::VectorTools::L2_norm)
            << std::endl;

      Vector<double> difference_per_cell(triangulation.n_active_cells());

      dealii::VectorTools::integrate_difference(mapping,
                                                dof_handler,
                                                approximate_solution,
                                                ExactSolution,
                                                difference_per_cell,
                                                qGauss,
                                                dealii::VectorTools::L1_norm);

      double h1_error = dealii::VectorTools::compute_global_error(triangulation,
                                                                  difference_per_cell,
                                                                  dealii::VectorTools::L1_norm);
      pcout << "L1 error = " << h1_error << std::endl;
    }

    std::vector<double>
    compute_volume_of_phases(const int              degree,
                             const int              n_q_points,
                             const DoFHandler<dim> &dof_handler,
                             const VectorType &     solution_levelset,
                             const double           time,
                             const MPI_Comm &       mpi_communicator,
                             const double           max_value = 1,
                             const double           min_value = -1.0)
    {
      FE_Q<dim>     fe(degree);
      FEValues<dim> fe_values(fe,
                              QGauss<dim>(n_q_points),
                              update_values | update_JxW_values | update_quadrature_points);

      std::vector<double> phi_at_q(QGauss<dim>(n_q_points).size());

      std::vector<double> volume_fraction;
      double              vol_phase_1 = 0;
      double              vol_phase_2 = 0;
      const double        threshhold  = 0.5;

      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            fe_values.get_function_values(solution_levelset,
                                          phi_at_q); // compute values of old solution

            for (const unsigned int q_index : fe_values.quadrature_point_indices())
              {
                const double phi_normalized =
                  UtilityFunctions::CharacteristicFunctions::normalize(phi_at_q[q_index],
                                                                       min_value,
                                                                       max_value);
                if (phi_normalized >= threshhold)
                  vol_phase_1 += fe_values.JxW(q_index);
                else
                  vol_phase_2 += fe_values.JxW(q_index);
              }
          }
      volume_fraction.emplace_back(Utilities::MPI::sum(vol_phase_1, mpi_communicator));
      volume_fraction.emplace_back(Utilities::MPI::sum(vol_phase_2, mpi_communicator));
      //@ todo: write template class for formatted output table
      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        std::cout << "vol phase 1: " << volume_fraction[0] << " vol phase 2: " << volume_fraction[1]
                  << std::endl;

      volume_table.add_value("time", time);
      volume_table.add_value("vol phase 1", volume_fraction[0]);
      volume_table.add_value("vol phase 2", volume_fraction[1]);

      return volume_fraction;
    }

    void
    collect_volume_fraction(const std::vector<double> &volume_fraction)
    {
      volumes.emplace_back(volume_fraction);
    }

    void
    print_volume_fraction_table(const MPI_Comm &mpi_communicator, const std::string filename)
    {
      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
          // @todo: improve that is written to a standard *txt-file
          std::ofstream out_file(filename);
          volume_table.write_tex(out_file);
          // size_t headerWidths[2] = {
          // std::string("time").size(),
          // std::string("volume phase 1").size()
          //};

          // if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
          //{
          // std::cout << "output file opened" << std::endl;
          // std::fstream fs;
          // fs.open (parameters.filename_volume_output, std::fstream::out);
          // fs.precision(10);
          //fs << "time | volume phase 1 | volume phase 2 " << std::endl;
          // fs << std::left << std::setw(headerWidths[0]) << time;
          // fs << "   " << std::left << std::setw(headerWidths[1]) << volume_fraction[0];
          // fs << "   " << std::left << std::setw(headerWidths[1]) << volume_fraction[1] <<
          // std::endl; fs.close();
        }
    }
  };
} // namespace MeltPoolDG
