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
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/utilities/timeiterator.hpp>
#include <meltpooldg/advection_diffusion/advection_diffusion_operation.hpp>

namespace MeltPoolDG
{
namespace AdvectionDiffusion
{
  using namespace dealii; 

  template <int dim>
  class AdvectionDiffusionProblem : public ProblemBase<dim>
  {
  private:
    using VectorType          = LinearAlgebra::distributed::Vector<double>;         
    using BlockVectorType     = LinearAlgebra::distributed::BlockVector<double>;         

  public:

    AdvectionDiffusionProblem() = default;
    
    void 
    run( std::shared_ptr<SimulationBase<dim>> base_in ) final
    {
      initialize(base_in);

      while ( !time_iterator.is_finished() )
      {
        const double dt = time_iterator.get_next_time_increment();
        scratch_data->get_pcout() << "t= " << std::setw(10) << std::left << time_iterator.get_current_time();
        advec_diff_operation.solve(dt);
        /*
         *  do paraview output if requested
         */
        output_results(time_iterator.get_current_time_step_number(),
                       base_in->parameters);
      }
    }

    std::string get_name() final { return "advection-diffusion problem"; };

  private:
    /*
     *  This function initials the relevant member data
     *  for the computation of the advection-diffusion problem
     */
    void 
    initialize( std::shared_ptr<SimulationBase<dim>> base_in )
    {
      /*
       *  setup scratch data
       */
      scratch_data = std::make_shared<ScratchData<dim>>();
      /*
       *  setup mapping
       */
      const auto mapping = MappingQGeneric<dim>(parameters.degree);
      scratch_data->set_mapping(mapping);
      /*
       *  setup DoFHandler
       */
      FE_Q<dim>    fe(parameters.degree);
      
      dof_handler.initialize(*base_in->triangulation, fe );
      IndexSet locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
      scratch_data->attach_dof_handler(dof_handler);

      /*
       *  make hanging nodes and dirichlet constraints (at the moment no time-dependent
       *  dirichlet constraints are supported)
       */
      constraints.clear();
      constraints.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      
      for (const auto& bc : base_in->get_boundary_conditions().dirichlet_bc) 
      {
        VectorTools::interpolate_boundary_values( dof_handler,
                                                  bc.first,
                                                  *bc.second,
                                                  constraints );
      }
      constraints.close();
      
      scratch_data->attach_constraint_matrix(constraints);
      /*
       *  create quadrature rule
       */
      QGauss<1> quad_1d_temp(parameters.degree+1) ; // evt. nicht mehr
      
      scratch_data->attach_quadrature(quad_1d_temp);
      /*
       *  create the matrix-free object
       */
      scratch_data->build();
      /*  
       *  initialize the time iterator
       */
      TimeIteratorData<double> time_data;
      time_data.start_time       = parameters.advec_diff.start_time;
      time_data.end_time         = parameters.advec_diff.end_time;
      time_data.time_increment   = parameters.advec_diff.time_step_size; 
      time_data.max_n_time_steps = 10000;
      
      time_iterator.initialize(time_data);
      
      /*
       *  set initial conditions of the levelset function
       */
      VectorType initial_solution;
      scratch_data->initialize_dof_vector(initial_solution);

      VectorTools::project( dof_handler, 
                            constraints,
                            scratch_data->get_quadrature(),
                            *base_in->get_field_conditions()->initial_field,           
                            initial_solution );

      initial_solution.update_ghost_values();

      /*
       *    initialize the advection-diffusion operation class
       */
      advection_velocity = base_in->get_advection_field();
      
      advec_diff_operation.initialize(scratch_data, 
                                      initial_solution, 
                                      parameters, 
                                      *advection_velocity);
    }

    void 
    output_results(const unsigned int time_step,
                   const Parameters<double>& parameters) 
    {
      if (parameters.paraview.do_output)
      {
        const MPI_Comm mpi_communicator = scratch_data->get_mpi_comm();
          
        /*
         *  output advected field
         */
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(advec_diff_operation.solution_advected_field, "advected_field");
        
        /*
         *  output advection velocity
         */
        BlockVectorType advection;
        scratch_data->initialize_block_dof_vector(advection);

        if ( parameters.paraview.print_advection )
        {
            advection_velocity->set_time( time_iterator.get_current_time() );
            /*
             *  work around to interpolate a vector-valued quantity on a scalar DoFHandler
             */
            for(auto d=0; d<dim; ++d)
            {
              VectorTools::interpolate(scratch_data->get_mapping(),
                                       scratch_data->get_dof_handler(),
                                       ScalarFunctionFromFunctionObject<dim>( [&](const Point<dim>& p) {return advection_velocity->value(p)[d];}),
                                       advection.block(d));
              advection.block(d).update_ghost_values();

              data_out.add_data_vector(dof_handler,advection.block(d), "advection_velocity_"+std::to_string(d));
            }
        }
        /*
        * write data to vtu file
        */
        data_out.build_patches();
        data_out.write_vtu_with_pvtu_record("./", "solution_advection_diffusion", time_step, mpi_communicator); // n_digits_timestep, n_groups);
        
        /*
        * write data of boundary -- @todo: move to own utility function
        */
        const unsigned int rank    = Utilities::MPI::this_mpi_process(mpi_communicator);
        const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(mpi_communicator);

        const unsigned int n_digits = static_cast<int>(std::ceil(std::log10(std::fabs(n_ranks))));

        std::string filename = "./solution_advection_diffusion_boundary_IDs" + Utilities::int_to_string(rank, n_digits) + ".vtk";
        std::ofstream output(filename.c_str());

        GridOut           grid_out;
        GridOutFlags::Vtk flags;
        flags.output_cells         = false;
        flags.output_faces         = true;
        flags.output_edges         = false;
        flags.output_only_relevant = false;
        grid_out.set_flags(flags);
        grid_out.write_vtk(scratch_data->get_dof_handler().get_triangulation(), output);
      
      }
    }
  private:
    DoFHandler<dim>                                      dof_handler;
    Parameters<double>                                   parameters; 
    AffineConstraints<double>                            constraints;    
    std::shared_ptr<ScratchData<dim>>                    scratch_data; 

    std::shared_ptr<TensorFunction<1,dim>>               advection_velocity;

    TimeIterator<double>                                 time_iterator;
    AdvectionDiffusionOperation<dim>             advec_diff_operation;
  };
} // namespace AdvectionDiffusion
} // namespace MeltPoolDG
