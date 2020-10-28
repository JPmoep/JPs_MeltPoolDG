/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
// enabling conditional ostreams
#include <deal.II/base/conditional_ostream.h>
// for index set
#include <deal.II/base/index_set.h>
//// for distributed triangulation
#include <deal.II/distributed/tria_base.h>
// for dof_handler type
#include <deal.II/dofs/dof_handler.h>
// for data_out
#include <deal.II/base/data_out_base.h>

#include <deal.II/numerics/data_out.h>
// for FE_Q<dim> type
#include <deal.II/fe/fe_q.h>
// for mapping
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_out.h>
// MeltPoolDG
#include <meltpooldg/interface/problembase.hpp>
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/level_set/level_set_operation.hpp>
#include <meltpooldg/utilities/timeiterator.hpp>
#include <meltpooldg/flow/adaflo_wrapper.hpp>
#include <meltpooldg/flow/adaflo_wrapper_parameters.hpp>

namespace MeltPoolDG
{
  namespace Flow
  {
    using namespace dealii;

    template <int dim>
    class TwoPhaseFlowProblem : public ProblemBase<dim>
    {
    private:
      using VectorType      = LinearAlgebra::distributed::Vector<double>;
      using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

    public:
      TwoPhaseFlowProblem() = default;

      void
      run(std::shared_ptr<SimulationBase<dim>> base_in) final
      {
        initialize(base_in);

        auto adaflo_params = base_in->parameters.adaflo_params;
        adaflo = std::make_shared<AdafloWrapper<dim>>(*scratch_data, 
                                                      adaflo_params);

        output_results(0,base_in->parameters);

        while (!time_iterator.is_finished())
        {
            const double dt = time_iterator.get_next_time_increment();
            adaflo->solve();
            output_results(time_iterator.get_current_time_step_number(), 
                           base_in->parameters);
        }
        
      }

      std::string
      get_name() final
      {
        return "two_phase_flow";
      };

    private:
      /*
       *  This function initials the relevant scratch data
       *  for the computation of the level set problem
       */
      void
      initialize(std::shared_ptr<SimulationBase<dim>> base_in)
      {
        /*
         *  setup scratch data
         */
        scratch_data =
          std::make_shared<ScratchData<dim>>(/*do_matrix_free*/ true);
        /*
         *  setup mapping
         */
        scratch_data->set_mapping(MappingQGeneric<dim>(base_in->parameters.base.degree));
          /*
           *  setup DoFHandler
           */
        dof_handler.initialize(*base_in->triangulation,
                                 FE_Q<dim>(base_in->parameters.base.degree));

        scratch_data->attach_dof_handler(dof_handler);

        time_iterator.initialize(TimeIteratorData<double>{0.0 /*start*/,
                                                    8 /*end*/,
                                                    0.02 /*dt*/,
                                                    1000 /*max_steps*/,
                                                    false /*cfl_condition-->not supported yet*/});


      }
      /*
       *  This function is to create paraview output
       */
      void
      output_results(const unsigned int time_step, const Parameters<double> &parameters) const
      {
        // if (parameters.paraview.do_output)
        // {
        //const MPI_Comm mpi_communicator = scratch_data->get_mpi_comm();
        /*
         *  output advected field
        */
        DataOut<dim> data_out;
        data_out.attach_dof_handler(scratch_data->get_dof_handler());

        data_out.add_data_vector(adaflo->get_velocity(), "velocity");

        data_out.build_patches(scratch_data->get_mapping());
        data_out.write_vtu_with_pvtu_record("./",
                                                parameters.paraview.filename,
                                                time_step,
                                                scratch_data->get_mpi_comm(),
                                                parameters.paraview.n_digits_timestep,
                                                parameters.paraview.n_groups);

        // }
      }
    private:
      TimeIterator<double> time_iterator;
      DoFHandler<dim>      dof_handler;

      std::shared_ptr<ScratchData<dim>> scratch_data;
      std::shared_ptr<AdafloWrapper<dim>> adaflo;
    };
  } // namespace TwoPhaseFlow
} // namespace MeltPoolDG