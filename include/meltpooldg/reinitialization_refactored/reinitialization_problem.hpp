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
// for FE_Q<dim> type
#include <deal.II/fe/fe_q.h>
// for mapping
#include <deal.II/fe/mapping.h>

// MeltPoolDG
#include <meltpooldg/interface/problembase.hpp>
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/utilities/timeiterator.hpp>
#include <meltpooldg/reinitialization_refactored/reinitialization_operation.hpp>

namespace MeltPoolDG
{
namespace ReinitializationNew
{
  using namespace dealii; 

  /*
   *     Reinitialization model for reobtaining the signed-distance 
   *     property of the level set equation
   */
  
  template <int dim, int degree>
  class ReinitializationProblem : public ProblemBase<dim,degree>
  {
  private:
    using VectorType          = LinearAlgebra::distributed::Vector<double>;         
    using DoFHandlerType      = DoFHandler<dim>;                                    
    using SparsityPatternType = TrilinosWrappers::SparsityPattern;

  public:

    /*
     *  Constructor of reinitialization problem
     */

    ReinitializationProblem( std::shared_ptr<SimulationBase<dim>> base_in )
    : fe(                  degree )
    , mapping(             degree )
    , q_gauss(             degree+1 )
    , triangulation(       base_in->triangulation_shared)
    , dof_handler(         *triangulation )
    , parameters(          base_in->parameters )
    , field_conditions(    base_in->get_field_conditions()  )
    , min_cell_size(       GridTools::minimal_cell_diameter(*triangulation) )
    , mpi_communicator(    base_in->get_mpi_communicator())
    , pcout(               std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    , reinit_operation(    dof_handler,
                           mapping,
                           fe,
                           q_gauss,
                           constraints,
                           locally_owned_dofs,
                           locally_relevant_dofs,
                           min_cell_size )
    {
    }

    /*
     *  This function is the global run function overriding the run() function from the ProblemBase
     *  class
     */

    void 
    run() final
    {
      initialize();
      while ( !time_iterator.is_finished() )
      {
        pcout << "t= " << std::setw(10) << std::left << time_iterator.get_current_time();
        reinit_operation.reinit_data.d_tau = time_iterator.get_next_time_increment();   
        reinit_operation.solve();
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
       *  make hanging nodes constraints
       */
      constraints.clear();
      constraints.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      constraints.close();
      
      /*  
       *  initialize the time iterator
       */
      TimeIteratorData time_data;
      time_data.start_time       = 0.0;
      time_data.end_time         = 100.;
      time_data.time_increment   = parameters.reinit_dtau; 
      time_data.max_n_time_steps = parameters.reinit_max_n_steps;
      
      time_iterator.initialize(time_data);
      
      /*
       *  set initial conditions of the levelset function
       */
      VectorType solution_levelset;
      solution_levelset.reinit( locally_owned_dofs, 
                                locally_relevant_dofs,
                                mpi_communicator);

      VectorTools::project( dof_handler, 
                            constraints,
                            q_gauss,
                            *field_conditions->initial_field,           
                            solution_levelset );

      solution_levelset.update_ghost_values();

      /*
       *    initialize the reinitialization operation class
       */
      reinit_operation.initialize(solution_levelset, parameters);
      
    }

    /*
     *  Creating paraview output
     */
    void 
    output_results(const unsigned int time_step=0) const
    {
      if (parameters.paraview_do_output)
      {
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(reinit_operation.solution_levelset, "psi");
        
        /*
         *  @todo: add_data_vector(exact_solution)
         */
        VectorType levelset_exact;
        levelset_exact.reinit( locally_owned_dofs,
                               mpi_communicator);
        data_out.build_patches();
      
        //const int n_digits_timestep = 4;
        //const int n_groups = 1;
        data_out.write_vtu_with_pvtu_record("./", "solution_reinitialization", time_step, mpi_communicator); // n_digits_timestep, n_groups);
      }
    }
  
    // for submodule this is actually needed as reference
    FE_Q<dim>                                            fe;
    MappingQGeneric<dim>                                 mapping;
    QGauss<dim>                                          q_gauss;
    std::shared_ptr<parallel::TriangulationBase<dim>>    triangulation;
    DoFHandlerType                                       dof_handler;
    Parameters<double>                                   parameters;
    std::shared_ptr<FieldConditions<dim>>                field_conditions;
    double                                               min_cell_size;     // @todo: check CFL condition
    const MPI_Comm                                       mpi_communicator;
    ConditionalOStream                                   pcout;
    /* 
    * at the moment the implementation considers natural boundary conditions
     */
    //std::shared_ptr<BoundaryConditions<dim>>   boundary_conditions;
    
    AffineConstraints<double>                            constraints;
    IndexSet                                             locally_owned_dofs;
    IndexSet                                             locally_relevant_dofs;
    
    TimeIterator                                         time_iterator;
    ReinitializationOperation<dim, degree>               reinit_operation;
  };
} // namespace Reinitialization
} // namespace MeltPoolDG
