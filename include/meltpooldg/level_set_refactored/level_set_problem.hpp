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
#include <meltpooldg/level_set_refactored/level_set_operation.hpp>

namespace MeltPoolDG
{
namespace LevelSet
{
  using namespace dealii; 

  /*
   *     Reinitialization model for reobtaining the signed-distance 
   *     property of the level set equation
   */
  
  template <int dim, int degree>
  class LevelSetProblem : public ProblemBase<dim,degree>
  {
  private:
    using VectorType          = LinearAlgebra::distributed::Vector<double>;         
    using BlockVectorType     = LinearAlgebra::distributed::BlockVector<double>;         
    using DoFHandlerType      = DoFHandler<dim>;                                    
    using SparsityPatternType = TrilinosWrappers::SparsityPattern;

  public:

    /*
     *  Constructor of advection-diffusion problem
     */

    LevelSetProblem( std::shared_ptr<SimulationBase<dim>> base_in )
    : fe(                      degree )
    , mapping(                 degree )
    , q_gauss(                 degree+1 )
    , triangulation(           base_in->triangulation)
    , dof_handler(             *triangulation)
    , parameters(              base_in->parameters )
    , field_conditions(        base_in->get_field_conditions()  )
    , boundary_conditions(     base_in->get_boundary_conditions()  )
    , min_cell_size(           GridTools::minimal_cell_diameter(*triangulation) )
    , mpi_communicator(        base_in->get_mpi_communicator())
    , pcout(                   std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    , level_set_operation(     dof_handler,
                               mapping,
                               fe,
                               q_gauss,
                               constraints,
                               locally_owned_dofs,                   //  @todo: locally_owned_dofs is not initialized
                               locally_relevant_dofs,
                               min_cell_size,
                               *field_conditions->advection_field,
                               constraints_no_dirichlet)
    {
    }
    
    void 
    run() final
    {
      initialize();

      while ( !time_iterator.is_finished() )
      {
        pcout << "| ls: t= " << std::setw(10) << std::left << time_iterator.get_current_time();
        level_set_operation.level_set_data.dt = time_iterator.get_next_time_increment();   
        level_set_operation.solve(parameters);
        /*
         *  do paraview output if requested
         */
        output_results(time_iterator.get_current_time_step_number());
      }
    }

    std::string get_name() final { return "reinitialization"; };

  private:
    /*
     *  This function initials the relevant member data
     *  for the computation of a reinitialization problem
     */
    void 
    initialize()
    {
      /*
       *  setup DoFHandler
       */
      dof_handler.distribute_dofs( fe );
      locally_owned_dofs = dof_handler.locally_owned_dofs(); 

      DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
      
      /*
       *  make hanging nodes and dirichlet constraints (at the moment no time-dependent
       *  dirichlet constraints are supported)
       */
      //constraints.emplace_back(AffineConstraints<double>());
      constraints.clear();
      constraints.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      for (const auto & bc : boundary_conditions->dirichlet_bc)
      {
        VectorTools::interpolate_boundary_values( dof_handler,
                                                  bc.first,
                                                  *bc.second,
                                                  constraints );
      }
      constraints.close();
      /*
       *  make hanging nodes 
       */
      constraints_no_dirichlet.clear();
      constraints_no_dirichlet.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler, constraints_no_dirichlet);
      constraints_no_dirichlet.close();
      /*  
       *  initialize the time iterator
       */
      time_iterator.initialize(TimeIteratorData{ parameters.ls_start_time,
                                                 parameters.ls_end_time,
                                                 parameters.ls_time_step_size,
                                                 1000,
                                                 false });
      /*
       *  set initial conditions of the levelset function
       */
      VectorType initial_solution;
      initial_solution.reinit( locally_owned_dofs, 
                                locally_relevant_dofs,
                                mpi_communicator);
      VectorTools::project( dof_handler, 
                            constraints,
                            q_gauss,
                            *field_conditions->initial_field,           
                            initial_solution );

      initial_solution.update_ghost_values();

      /*
       *    initialize the reinitialization operation class
       */
      level_set_operation.initialize(initial_solution, parameters);
      
    }
    /*
     *  This function is to create paraview output
     */
    void 
    output_results(const unsigned int time_step=0) const
    {
      if (parameters.paraview_do_output)
      {
        /*
         *  output advected field
         */
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(level_set_operation.solution_level_set, "level_set");
        
        /*
         *  output advection velocity
         *  @ todo --> clean up 
         */
        
        BlockVectorType advection;
        advection.reinit(dim);
        for(auto d=0; d<dim; ++d)
          advection.block(d).reinit(dof_handler.n_dofs() ); 
        
        if (parameters.paraview_print_advection)
        {
          field_conditions->advection_field->set_time( time_iterator.get_current_time() );
          std::map<types::global_dof_index, Point<dim> > supportPoints;
          DoFTools::map_dofs_to_support_points<dim,dim>(mapping,dof_handler,supportPoints);

          for(auto& global_dof : supportPoints)
          {
              auto a = field_conditions->advection_field->value(global_dof.second);
              for(auto d=0; d<dim; ++d)
                advection.block(d)[global_dof.first] = a[d];
          }


          for(auto d=0; d<dim; ++d)
            data_out.add_data_vector(dof_handler,advection.block(d), "advection_velocity_"+std::to_string(d));
        }
        /*
        * write data to vtu file
        */
        data_out.build_patches();
        data_out.write_vtu_with_pvtu_record("./", "solution_levelset", time_step, mpi_communicator, 4, 1); 
        
        /*
        * write data of boundary -- @todo: move to own utility function
        */
        if (parameters.paraview_print_boundary_id)
        {
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
          grid_out.write_vtk(*triangulation, output);
        }
      }
    }
    
    FE_Q<dim>                                            fe;
    MappingQGeneric<dim>                                 mapping;
    QGauss<dim>                                          q_gauss;
    std::shared_ptr<parallel::TriangulationBase<dim>>    triangulation;
    DoFHandlerType                                       dof_handler;
    Parameters<double>                                   parameters;
    std::shared_ptr<FieldConditions<dim>>                field_conditions;
    std::shared_ptr<BoundaryConditions<dim>>             boundary_conditions;
    
    double                                               min_cell_size;     // @todo: check CFL condition
    const MPI_Comm                                       mpi_communicator;
    ConditionalOStream                                   pcout;
    
    AffineConstraints<double>                            constraints;
    AffineConstraints<double>                            constraints_no_dirichlet;
    //std::vector<AffineConstraints<double>>               constraints_vec; // @todo
    IndexSet                                             locally_owned_dofs;
    IndexSet                                             locally_relevant_dofs;
    
    TimeIterator                                         time_iterator;
    LevelSetOperation<dim, degree>                       level_set_operation;
  };
} // namespace Reinitialization
} // namespace MeltPoolDG
