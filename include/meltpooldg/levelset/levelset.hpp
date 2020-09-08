/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, August 2020
 *
 * ---------------------------------------------------------------------
 */
#pragma once

#include <deal.II/base/timer.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

// for distributed vectors/matrices
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>

// sparse matrices utilites
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

// (non)-distributed algebra
#include <deal.II/lac/full_matrix.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

// solvers
#include <deal.II/lac/solver_cg.h> // only for symmetric matrices
#include <deal.II/lac/solver_gmres.h>

// preconditioner
#include <deal.II/lac/precondition.h>

// constraints
#include <deal.II/lac/affine_constraints.h>

// grid-specific libraries
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

// dof handler
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

// finite element
#include <deal.II/fe/fe_q.h>

// additional includes for parallelization
#include <deal.II/base/conditional_ostream.h> 
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h> 
#include <deal.II/base/utilities.h>
#include <deal.II/base/data_out_base.h>

// c++
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm> // for enum ::All
#include <cmath>

// meltpooldg
#include <meltpooldg/reinitialization/reinitialization.hpp>
#include <meltpooldg/curvature/curvature.hpp>
#include <meltpooldg/utilities/utilityfunctions.hpp>
#include <meltpooldg/interface/problembase.hpp>
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/interface/parameters.hpp>

namespace MeltPoolDG
{
  using namespace dealii; 

  template <int dim, int degree>
  class LevelSetEquation : public ProblemBase<dim,degree>
  {
  private:
    using VectorType          = LinearAlgebra::distributed::Vector<double>;         
    using BlockVectorType     = LinearAlgebra::distributed::BlockVector<double>;    
    using SparseMatrixType    = TrilinosWrappers::SparseMatrix;                     
    using DoFHandlerType      = DoFHandler<dim>;                                    
    using SparsityPatternType = TrilinosWrappers::SparsityPattern;
    using ConstraintsType     = AffineConstraints<double>;

  public:
    LevelSetEquation( std::shared_ptr<SimulationBase<dim>> base );
    
    void run() final;
    
    std::string get_name() final { return "levelset"; };

  private:
    void 
    initialize_module();
    /*
     *      set the initial conditions for the level set equation
     */
    void 
    set_initial_conditions(); 
    /*
     *      initialize the time iterator for solving the level set problem
     *      with the given input parameters
     */
    void
    initialize_time_iterator();
    /*
     *      solve level set equation
     */
    void 
    compute_levelset_model();
    /*
     *      setup reinitialization model
     */
    void 
    initialize_reinitialization_model();
    /*
     *      solve reinitialization model
     */
    void 
    compute_reinitialization_model();
    /*
     *      initialize the curvature calculation routine
     */
    
    void 
    initialize_curvature();
    /*
     *      compute the curvature for given level set solution vector
     */
    void 
    compute_curvature();

    void output_results(double timestep=-1.0);
    void print_me();
    
    MPI_Comm                                   mpi_communicator;
    Parameters<double>                         parameters;
    FE_Q<dim>                                  fe;                         // @todo: should it stay a member variable?
    parallel::distributed::Triangulation<dim>& triangulation;
    DoFHandler<dim>                            dof_handler;
    ConditionalOStream                         pcout;
    TimerOutput                                computing_timer;
    Timer                                      timer;
    std::shared_ptr<FieldConditions<dim>>      field_conditions;
    std::shared_ptr<BoundaryConditions<dim>>   boundary_conditions;
    /* 
     * the following are subproblem objects
    */
    Reinitialization<dim,degree>               reini;
    Curvature<dim,degree>                      curvature;

    AffineConstraints<double>                  constraints;
    AffineConstraints<double>                  constraints_no_dirichlet;

    SparseMatrixType                           system_matrix;              
    VectorType                                 system_rhs;                
    VectorType                                 solution_levelset;

    IndexSet                                   locally_owned_dofs;
    IndexSet                                   locally_relevant_dofs;

    TimeIterator                               time_iterator;

  };
} // namespace MeltPoolDG
