/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, August 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
// enabling conditional ostreams
#include <deal.II/base/conditional_ostream.h> 
// for index set
#include <deal.II/base/index_set.h>
// for mpi
#include <deal.II/base/mpi.h> 
// for quadrature points
#include <deal.II/base/quadrature_lib.h>
// for using smart pointers
#include <deal.II/base/smartpointer.h>
//// for distributed triangulation
//#include <deal.II/distributed/tria.h>
// for dof_handler type
#include <deal.II/dofs/dof_handler.h>
// for FE_Q<dim> type
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/distributed/tria_base.h>

// MeltPoolDG
#include <meltpooldg/utilities/utilityfunctions.hpp>
#include <meltpooldg/utilities/timeiterator.hpp>
#include <meltpooldg/interface/problembase.hpp>
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/normal_vector/normalvector.hpp>
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
    using BlockVectorType     = LinearAlgebra::distributed::BlockVector<double>;    
    using SparseMatrixType    = TrilinosWrappers::SparseMatrix;                     
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
    run()
    {
      initialize();
      
      while ( !time_iterator.is_finished() )
      {
        pcout << "t= " << time_iterator.get_current_time() << std::endl;
        reinit_operation.reinit_data.d_tau = time_iterator.get_next_time_increment();   
        reinit_operation.solve();
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
      
      /*  
       *  initialize the time iterator
       */
      TimeIteratorData time_data;
      time_data.start_time       = 0.0;
      time_data.end_time         = 100.;
      time_data.time_increment   = parameters.reinit_dtau; 
      time_data.max_n_time_steps = parameters.reinit_max_n_steps;
      
      time_iterator.initialize(time_data);
    }

    /*
     *  Creating paraview output
     */
    void 
    output_results(const double time=0.0)
    {
      if (parameters.paraview_do_output)
      {
        DataOut<dim> data_out;
        data_out.attach_dof_handler(*dof_handler);
        data_out.add_data_vector(reinit_operation.solution_levelset, "phi");

        VectorType levelset_exact;
        levelset_exact.reinit( locally_owned_dofs,
                                mpi_communicator);
        data_out.build_patches();
      
        const int n_digits_timestep = 2;
        const int n_groups = 1;
        data_out.write_vtu_with_pvtu_record("./", "solution_reinitialization", time, mpi_communicator, n_digits_timestep, n_groups);
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
    
    //VectorType                                           solution_levelset;        // @todo: might not be member variables
    TimeIterator                                         time_iterator;
    ReinitializationOperation<dim, degree>               reinit_operation;
  };
} // namespace ReinitializationNew
} // namespace MeltPoolDG
