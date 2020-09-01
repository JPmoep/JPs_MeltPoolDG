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
  class LevelSetEquation
  {
  private:
    typedef LinearAlgebra::distributed::Vector<double>         VectorType;
    typedef LinearAlgebra::distributed::BlockVector<double>    BlockVectorType;
    typedef TrilinosWrappers::SparseMatrix                     SparseMatrixType;

  public:
    LevelSetEquation(
                     //parallel::distributed::Triangulation<dim>&  triangulation,
                     std::shared_ptr<SimulationBase<dim>>        base
                     //MPI_Comm&                                   mpi_commun
                     );
    void run( );
    void compute_error( const Function<dim>& ExactSolution );
    double epsilon;

  private:
    void setup_system();
    /*
     *      initialize level set equation
     */
    void initialize_levelset( ); 
    /*
     *      solve level set equation
     */
    void assemble_levelset_system( );
    /*
     *      setup reinitialization model
     */
    void initialize_reinitialization_model();
    /*
     *      solve reinitialization model
     */
    void compute_reinitialization_model();
    /*
     *      initialize the curvature calculation routine
     */
    void initialize_curvature();
    /*
     *      compute the curvature for given level set solution vector
     */
    void compute_curvature();
    
    void solve_u();
    
    void output_results(const double timeStep);
           //TensorFunction<1, dim> &AdvectionField_ );
    void print_me();

    void compute_overall_phase_volume();
    void computeAdvection(); //TensorFunction<1, dim> &AdvectionField_);
    
    MPI_Comm&                                  mpi_communicator;
    LevelSetParameters                         parameters;
    FE_Q<dim>                                  fe;
    parallel::distributed::Triangulation<dim>& triangulation;
    DoFHandler<dim>                            dof_handler;
    QGauss<dim>                                qGauss; 
    double                                     time_step;
    double                                     time;
    unsigned int                               timestep_number;
    
    AffineConstraints<double>                  constraints;
    AffineConstraints<double>                  constraints_no_dirichlet;

    SparseMatrixType                           systemMatrix;              // global system matrix
    VectorType                                 systemRHS;                 // global system right-hand side
    VectorType                                 solution_u;
    
    std::vector<double>                        volume_fraction;
    IndexSet                                   locally_owned_dofs;
    IndexSet                                   locally_relevant_dofs;
    ConditionalOStream                         pcout;
    TimerOutput                                computing_timer;
    Timer                                      timer;
    //TensorFunction<1, dim> &                   AdvectionField;

    std::shared_ptr<FieldConditions<dim>>      field_conditions;
    std::shared_ptr<BoundaryConditions<dim>>   boundary_conditions;

    /* 
     * the following are subproblem classes
    */
    Reinitialization<dim,degree>              reini;
    Curvature<dim,degree>                     curvature;
  };
} // end of namespace LevelSet
