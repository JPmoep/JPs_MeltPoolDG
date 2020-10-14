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
#include <meltpooldg/utilities/timeiterator.hpp>
#include <meltpooldg/level_set/level_set_operation.hpp>

namespace MeltPoolDG
{
namespace LevelSet
{
  using namespace dealii; 

  template <int dim>
  class LevelSetProblem : public ProblemBase<dim>
  {
  private:
    using VectorType          = LinearAlgebra::distributed::Vector<double>;         
    using BlockVectorType     = LinearAlgebra::distributed::BlockVector<double>;         

  public:

    LevelSetProblem() = default;
    
    void 
    run( std::shared_ptr<SimulationBase<dim>> base_in ) final
    {
      initialize(base_in);
      
      while ( !time_iterator.is_finished() )
      {
        const double dt = time_iterator.get_next_time_increment();   
        scratch_data->get_pcout() << "| ls: t= " << std::setw(10) << std::left << time_iterator.get_current_time();
        level_set_operation.solve(dt);
        /*
         *  do paraview output if requested
         */
        output_results(time_iterator.get_current_time_step_number(),
                       base_in->parameters);
      }
    }

    std::string get_name() final { return "level_set_problem"; };

  private:
    /*
     *  This function initials the relevant scratch data
     *  for the computation of the level set problem
     */
    void 
    initialize(std::shared_ptr<SimulationBase<dim>> base_in )
    {
      /*
       *  setup scratch data
       */
      scratch_data = std::make_shared<ScratchData<dim>>();
      /*
       *  setup mapping
       */
      auto mapping = MappingQGeneric<dim>(base_in->parameters.base.degree);
      scratch_data->set_mapping(mapping);
      /*
       *  setup DoFHandler
       */
      FE_Q<dim>    fe(base_in->parameters.base.degree);
      
      dof_handler.initialize(*base_in->triangulation, fe );
      scratch_data->attach_dof_handler(dof_handler);
      scratch_data->attach_dof_handler(dof_handler);
      /*
       *  create partitioning
       */
      scratch_data->create_partitioning();
      /*
       *  make hanging nodes constraints
       */
      constraints.clear();
      constraints.reinit(scratch_data->get_locally_relevant_dofs());
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      constraints.close();

      scratch_data->attach_constraint_matrix(constraints);
     
      /*
       *  make hanging nodes and dirichlet constraints (at the moment no time-dependent
       *  dirichlet constraints are supported)
       */
      constraints_dirichlet.clear();
      constraints_dirichlet.reinit(scratch_data->get_locally_relevant_dofs());
      DoFTools::make_hanging_node_constraints(dof_handler, constraints_dirichlet);
      for (const auto& bc : base_in->get_boundary_conditions().dirichlet_bc) 
      {
        VectorTools::interpolate_boundary_values( dof_handler,
                                                  bc.first,
                                                  *bc.second,
                                                  constraints_dirichlet );
      }
      constraints_dirichlet.close(); 

      scratch_data->attach_constraint_matrix(constraints_dirichlet);
      /*
       *  create quadrature rule
       */
      QGauss<1> quad_1d_temp(base_in->parameters.base.n_q_points_1d); 
      
      scratch_data->attach_quadrature(quad_1d_temp);
      scratch_data->attach_quadrature(quad_1d_temp);
      /*
       *  create the matrix-free object
       */
      scratch_data->build();
      /*  
       *  initialize the time iterator
       */
      time_iterator.initialize(TimeIteratorData<double>{ base_in->parameters.ls.start_time,
                                                 base_in->parameters.ls.end_time,
                                                 base_in->parameters.ls.time_step_size,
                                                 100000,
                                                 false });
      /*
       *  set initial conditions of the levelset function
       */
      VectorType initial_solution;
      scratch_data->initialize_dof_vector(initial_solution);
      VectorTools::project( dof_handler, 
                            constraints_dirichlet,
                            scratch_data->get_quadrature(),
                            *base_in->get_field_conditions()->initial_field,
                            initial_solution );

      initial_solution.update_ghost_values();

      /*
       *    initialize the levelset operation class
       */
      advection_velocity = base_in->get_advection_field();
      AssertThrow(base_in->get_advection_field(), 
                  ExcMessage(" It seems that your SimulationBase object does not contain "
                             "a valid advection velocity. A shared_ptr to your advection velocity "
                             "function, e.g., AdvectionFunc<dim> must be specified as follows: "
                             "this->field_conditions.advection_field = std::make_shared<AdvectionFunc<dim>>();" 
                            ));
      
      level_set_operation.initialize(scratch_data, 
                                     initial_solution, 
                                     base_in->parameters, 
                                     advection_velocity);
    }

    /*
     *  This function is to create paraview output
     */
    void 
    output_results(const unsigned int time_step,
                   const Parameters<double>& parameters) const
    {
      if (parameters.paraview.do_output)
      {
        const MPI_Comm mpi_communicator = scratch_data->get_mpi_comm();
        /*
         *  output advected field
         */
        DataOut<dim> data_out;
        data_out.attach_dof_handler(scratch_data->get_dof_handler());
        data_out.add_data_vector(level_set_operation.solution_level_set, "level_set");
        
        /*
         *  output normal vector field
         */
        if (parameters.paraview.print_normal_vector)
          for (unsigned int d=0; d<dim; ++d)
            data_out.add_data_vector(level_set_operation.solution_normal_vector.block(d), "normal_"+std::to_string(d));
        
        /*
         *  output advection velocity
         */
        BlockVectorType advection;
        scratch_data->initialize_dof_vector(advection);

        if (parameters.paraview.print_advection)
        {
          advection_velocity->set_time(time_iterator.get_current_time());
          /*
           *  work around to interpolate a vector-valued quantity on a scalar DoFHandler
           */
          for (auto d = 0; d < dim; ++d)
            {
              VectorTools::interpolate(scratch_data->get_mapping(),
                                       scratch_data->get_dof_handler(),
                                       ScalarFunctionFromFunctionObject<dim>(
                                         [&](const Point<dim> &p) {
                                           return advection_velocity->value(p)[d];
                                         }),
                                       advection.block(d));
              advection.block(d).update_ghost_values();

              data_out.add_data_vector(dof_handler,
                                       advection.block(d),
                                       "advection_velocity_" + std::to_string(d));
            }
        }
        /*
        * write data to vtu file
        */
        data_out.build_patches();
        data_out.write_vtu_with_pvtu_record("./", 
                                            parameters.paraview.filename, 
                                            time_step, 
                                            scratch_data->get_mpi_comm(), 
                                            parameters.paraview.n_digits_timestep, 
                                            parameters.paraview.n_groups);
        
        /*
        * write data of boundary -- @todo: move to own utility function
        */
        if (parameters.paraview.print_boundary_id)
        {
          const unsigned int rank    = Utilities::MPI::this_mpi_process(mpi_communicator);
          const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(mpi_communicator);

          const unsigned int n_digits = static_cast<int>(std::ceil(std::log10(std::fabs(n_ranks))));

          std::string filename = "./solution_level_set_boundary_IDs" + Utilities::int_to_string(rank, n_digits) + ".vtk";
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
    }
  private:
    DoFHandler<dim>                                      dof_handler;
    AffineConstraints<double>                            constraints_dirichlet;
    AffineConstraints<double>                            constraints;
    
    std::shared_ptr<ScratchData<dim>>                    scratch_data; 
    std::shared_ptr<TensorFunction<1,dim>>               advection_velocity;
    
    TimeIterator<double>                                 time_iterator;
    LevelSetOperation<dim>                               level_set_operation;
  };
} // namespace LevelSet
} // namespace MeltPoolDG
