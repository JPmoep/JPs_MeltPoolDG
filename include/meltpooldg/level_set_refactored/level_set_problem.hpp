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
     *  Constructor of the levelset problem
     */
    LevelSetProblem()
    {
    }
    
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
    initialize(std::shared_ptr<SimulationBase<dim>> base_in )
    {
      /*
       *  setup scratch data
       */
      scratch_data = std::make_shared<ScratchData<dim>>();
      /*
       *  setup mapping
       */
      auto mapping = MappingQGeneric<dim>(degree);
      scratch_data->set_mapping(mapping);
      /*
       *  setup DoFHandler
       */
      FE_Q<dim>    fe(degree);
      
      dof_handler.initialize(*base_in->triangulation, fe );
      IndexSet locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
      scratch_data->attach_dof_handler(dof_handler);
      scratch_data->attach_dof_handler(dof_handler);

      /*
       *  make hanging nodes constraints
       */
      constraints.clear();
      constraints.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      constraints.close();

      scratch_data->attach_constraint_matrix(constraints);
     
      /*
       *  make hanging nodes and dirichlet constraints (at the moment no time-dependent
       *  dirichlet constraints are supported)
       */
      constraints_dirichlet.clear();
      constraints_dirichlet.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler, constraints_dirichlet);
      //
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
      QGauss<1> quad_1d_temp(degree+1) ; // evt. nicht mehr
      
      scratch_data->attach_quadrature(quad_1d_temp);
      scratch_data->attach_quadrature(quad_1d_temp);
      /*
       *  create the matrix-free object
       */
      scratch_data->build();
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
      level_set_operation.initialize(scratch_data, 
                                      initial_solution, 
                                      parameters, 
                                      advection_velocity);
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
        //DataOut<dim> data_out;
        //data_out.attach_dof_handler(matrix_free.get_dof_handler());
        //data_out.add_data_vector(level_set_operation.solution_level_set, "level_set");
        
        /*
         *  output advection velocity
         *  @ todo --> clean up 
         */
        
        //BlockVectorType advection;
        //advection.reinit(dim);
        //for(auto d=0; d<dim; ++d)
          //advection.block(d).reinit(matrix_free.get_dof_handler().n_dofs() ); 
        
        //if (parameters.paraview_print_advection)
        //{
          //field_conditions->advection_field->set_time( time_iterator.get_current_time() );
          //std::map<types::global_dof_index, Point<dim> > supportPoints;
          //DoFTools::map_dofs_to_support_points<dim,dim>(mapping,
                                                        //matrix_free.get_dof_handler(),
                                                        //supportPoints);
          //for(auto& global_dof : supportPoints)
          //{
              //auto a = field_conditions->advection_field->value(global_dof.second);
              //for(auto d=0; d<dim; ++d)
                //advection.block(d)[global_dof.first] = a[d];
          //}


          //for(auto d=0; d<dim; ++d)
            //data_out.add_data_vector(matrix_free.get_dof_handler(),
                                     //advection.block(d), 
                                     //"advection_velocity_"+std::to_string(d));
        //}
        /*
        * write data to vtu file
        */
        //data_out.build_patches();
        //data_out.write_vtu_with_pvtu_record("./", "solution_levelset", time_step, mpi_communicator, 4, 1); 
        
        /*
        * write data of boundary -- @todo: move to own utility function
        */
        //if (parameters.paraview_print_boundary_id)
        //{
          //const unsigned int rank    = Utilities::MPI::this_mpi_process(mpi_communicator);
          //const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(mpi_communicator);

          //const unsigned int n_digits = static_cast<int>(std::ceil(std::log10(std::fabs(n_ranks))));

          //std::string filename = "./solution_levelset_boundary_IDs" + Utilities::int_to_string(rank, n_digits) + ".vtk";
          //std::ofstream output(filename.c_str());

          //GridOut           grid_out;
          //GridOutFlags::Vtk flags;
          //flags.output_cells         = false;
          //flags.output_faces         = true;
          //flags.output_edges         = false;
          //flags.output_only_relevant = false;
          //grid_out.set_flags(flags);
          //grid_out.write_vtk(*triangulation, output);
        //}
      }
    }
  private:
    DoFHandler<dim>                                      dof_handler;
    Parameters<double>                                   parameters; //evt. nicht mehr
    AffineConstraints<double>                            constraints_dirichlet;
    AffineConstraints<double>                            constraints;
    
    std::shared_ptr<ScratchData<dim>>                    scratch_data; 
    std::shared_ptr<TensorFunction<1,dim>>                advection_velocity;
    
    TimeIterator                                         time_iterator;
    LevelSetOperation<dim, degree>                       level_set_operation;
  };
} // namespace Reinitialization
} // namespace MeltPoolDG
