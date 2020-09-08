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
// MeltPoolDG
#include <meltpooldg/utilities/utilityfunctions.hpp>
#include <meltpooldg/normal_vector/normalvector.hpp>
#include <meltpooldg/utilities/timeiterator.hpp>
#include <meltpooldg/interface/problembase.hpp>
#include <meltpooldg/interface/simulationbase.hpp>

namespace MeltPoolDG
{
namespace Reinitialization
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
    using ConstraintsType     = AffineConstraints<double>;   

  public:

    /*
     *  Constructor as main module
     */
    Reinitialization( std::shared_ptr<SimulationBase<dim>> base )
    : mpi_communicator(    base_in->get_mpi_communicator())
    , pcout(               std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    , fe(                  degree )
    , mapping(             degree )
    , q_gauss(             degree+1 )
    , triangulation(       base_in->triangulation_shared)
    , dof_handler(         base_in->triangulation_shared)
    , field_conditions(    base_in->get_field_conditions()  )
    , min_cell_size(       GridTools::minimal_cell_diameter(base_in->triangulation_shared) )
    {
    }

    /*
     *  Usage as module: this function is the "global" run function to be called from the problem base class
     */
    void 
    run()
    {
      ReinitializationOperation<dim,degree> reinit_operation;
      reinit_operation.initialize();
      while ( !time_iterator.is_finished() )
      {
        const double d_tau = time_iterator.get_next_time_increment();   
        reinit_operation.solve(solution_levelset);
      }
    
    }

    std::string get_name() final { return "reinitialization"; };

  private:
    /*
     *  Usage as module: this function initials the relevant member data
     *  for the computation of the module
     */
    void 
    initialize()
    {
      module_dof_handler.distribute_dofs( FE_Q<dim>(degree) );
      locally_owned_dofs = module_dof_handler.locally_owned_dofs(); 
      DoFTools::extract_locally_relevant_dofs(module_dof_handler, locally_relevant_dofs);
      
      constraints.clear();
      constraints.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      module_constraints.close();

      solution_levelset.reinit( locally_owned_dofs, 
                                locally_relevant_dofs,
                                mpi_communicator);

      VectorTools::project( dof_handler, 
                            constraints,
                            QGauss<dim>(degree+1),
                            *field_conditions->initial_field,           
                            solution_levelset );

      solution_levelset.update_ghost_values();

      /*  
       *  initialize the time iterator
       */
      TimeIteratorData time_data;
      time_data.start_time       = 0.0;
      time_data.end_time         = 100.;
      time_data.time_increment   = reinit_data.d_tau; 
      time_data.max_n_time_steps = reinit_data.max_reinit_steps;
      
      t.initialize(time_data);
    }

    /*
     *  Usage as module: this function is for creating paraview output
     */
    void 
    output_results(const VectorType& solution, const double time=0.0)
    {
      if (parameters.paraview_do_output)
      {
        DataOut<dim> data_out;
        data_out.attach_dof_handler(*dof_handler);
        data_out.add_data_vector(solution_in, "phi");

        VectorType levelset_exact;
        levelset_exact.reinit( locally_owned_dofs,
                                mpi_communicator);
        data_out.build_patches();
      
        const int n_digits_timestep = 2;
        const int n_groups = 1;
        data_out.write_vtu_with_pvtu_record("./", "solution_reinitialization", time, mpi_communicator, n_digits_timestep, n_groups);
      }
    }
    /* 
     * This function solves the Olsson, Kreiss, Zahedi (2007) model for reinitialization 
     * of the level set equation.
     */
  
    // for submodule this is actually needed as reference
    const MPI_Comm&                            mpi_communicator;
    ConditionalOStream                         pcout;
    FE_Q<dim>                                  fe;
    MappingQGeneric<dim>                       mapping;
    QGauss<dim>                                q_gauss;
    std::shared_ptr<TriangulationBase<dim>>    triangulation;
    DoFHandlerType                             dof_handler;
    std::shared_ptr<FieldConditions<dim>>      field_conditions;
    double                                     min_cell_size;     // @todo: check CFL condition
    
    ConstraintsType                            constraints;
    Parameters<double>                         parameters;
    /* 
    * at the moment the implementation considers natural boundary conditions
     */
    //std::shared_ptr<BoundaryConditions<dim>>   boundary_conditions;
    
    IndexSet                                   locally_owned_dofs;
    IndexSet                                   locally_relevant_dofs;
  
    VectorType                                 solution_levelset;        // @todo: might not be member variables
    TimeIterator                               t;
  };
} // namespace Reinitialization
} // namespace MeltPoolDG
