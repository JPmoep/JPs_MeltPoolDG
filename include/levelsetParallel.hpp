/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2006 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, 2020
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

// multiphaseflow
#include "reinitialization.hpp"
#include "curvature.hpp"
#include "levelsetparameters.hpp"
#include "utilityFunctions.hpp"
#include "problembase.hpp"
#include "simulationbase.hpp"
// c++
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm> // for enum ::All
#include <cmath>

namespace LevelSetParallel
{
  using namespace dealii; 

  template <int dim, int degree>
  class LevelSetEquation : public ProblemBase<dim>
  {
  private:
    typedef LinearAlgebra::distributed::Vector<double>         VectorType;
    typedef LinearAlgebra::distributed::BlockVector<double>    BlockVectorType;
    typedef TrilinosWrappers::SparseMatrix                     SparseMatrixType;

  public:
    LevelSetEquation( std::shared_ptr<SimulationBase<dim>> base );
    void run() override;

  private:
    void 
    compute_error( const Function<dim>& ExactSolution );
    
    void 
    setup_system();
    /*
     *      initialize level set equation
     */
    void 
    initialize_levelset(); 
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
    compute_levelset_model( );
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
    LevelSetParameters                         parameters;
    FE_Q<dim>                                  fe;                         // @todo: should it stay a member variable?
    parallel::distributed::Triangulation<dim>& triangulation;
    DoFHandler<dim>                            dof_handler;
    AffineConstraints<double>                  constraints;
    AffineConstraints<double>                  constraints_no_dirichlet;

    SparseMatrixType                           system_matrix;              // global system matrix
    VectorType                                 system_rhs;                 // global system right-hand side
    VectorType                                 solution_levelset;

    IndexSet                                   locally_owned_dofs;
    IndexSet                                   locally_relevant_dofs;
    ConditionalOStream                         pcout;
    TimerOutput                                computing_timer;
    Timer                                      timer;

    TimeIterator                               time_iterator;

    std::shared_ptr<FieldConditions<dim>>      field_conditions;
    std::shared_ptr<BoundaryConditions<dim>>   boundary_conditions;

    /* 
     * the following are subproblem objects
    */
    Reinitialization<dim,degree>               reini;
    Curvature<dim,degree>                      curvature;
  };
} // end of namespace LevelSet
