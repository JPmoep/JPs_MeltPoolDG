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
#include <deal.II/lac/generic_linear_algebra.h>
namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>

#include <deal.II/fe/mapping.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
//#include <deal.II/lac/sparse_matrix.h>
//#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h> // only for symmetric matrices
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

//#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/utilities.h>
#include <deal.II/matrix_free/fe_evaluation.h>
//#include <experimental/filesystem>
#include "levelsetparameters.hpp"
#include <fstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm> // for enum ::All
#include <cmath>
#include "utilityFunctions.hpp"

// additional includes for parallelization
#include <deal.II/base/conditional_ostream.h> 
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/base/mpi.h> 
//#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
//#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/base/hdf5.h>

#include <deal.II/base/data_out_base.h>
#include <deal.II/lac/trilinos_solver.h> // direct solver

// multiphaseflow
#include <reinitialization.hpp>

namespace LevelSetParallel
{
  using namespace dealii; 

  template <int dim>
  class LevelSetEquation
  {
  public:
    LevelSetEquation(const LevelSetParameters& parameters,
                     parallel::distributed::Triangulation<dim>&       triangulation,
                     TensorFunction<1, dim>& AdvectionField_,
                     MPI_Comm& mpi_commun);
    void run( const Function<dim>& InitialValues,
              const Function<dim>& DirichletValues);
    void compute_error( const Function<dim>& ExactSolution );
    double epsilon;

  private:
    void setup_system(const Function<dim>& DirichletValues );
    void setInitialConditions(const Function<dim>& InitialValues);
    void setInitialConditions_reinitialize(const Function<dim>& InitialValues);
    void assemble_levelset_system( const Function<dim>& DirichletValues );


    /*
     *      setup reinitialization model
     */
    void initialize_reinitialization_model();
    /*
     *      solve reinitialization model
     */
    void compute_reinitialization_model();
    /*
     *      initialize normal vector routine
     */
    void initialize_normal_vectors();
    /*
     *      compute normal vecotr for given level set solution vector
     */
    void compute_normal_vectors();
    /*
     *      initialize normal vector routine
     */
    void initialize_curvature();
    /*
     *      compute normal vector for given level set solution vector
     */
    void compute_curvature();
    /*
     *      compute normal vector for given level set solution vector
     */
    void solve_u();
    void solve_cg(const LA::MPI::Vector& RHS,
                  const LA::MPI::SparseMatrix& matrix,
                  LA::MPI::Vector& solution,
                  const std::string& callerFunction);
    void output_results(const double timeStep);
           //TensorFunction<1, dim> &AdvectionField_ );
    void compute_overall_phase_volume();
    void computeAdvection(TensorFunction<1, dim> &AdvectionField_);
    void computeNormalLevelSet();
    void computeCurvatureLevelSet();
    MPI_Comm&                 mpi_communicator;
    LevelSetParameters        parameters;
    FE_Q<dim>                 fe;
    parallel::distributed::Triangulation<dim>& triangulation;
    DoFHandler<dim>           dof_handler;
    QGauss<dim>               qGauss; 
    double                    time_step;
    double                    time;
    unsigned int              timestep_number;
    
    AffineConstraints<double> constraints;
    AffineConstraints<double> constraints_re;

    LA::MPI::SparseMatrix      systemMatrix;              // global system matrix
    LA::MPI::SparseMatrix      systemMatrix_re;              // global system matrix
    LA::MPI::Vector            systemRHS;                 // global system right-hand side
    LA::MPI::Vector            solution_u;
    
    LA::MPI::Vector            re_solution_u;
    LA::MPI::Vector            re_delta_solution_u;
    LA::MPI::BlockVector       normal_vector_field;
    LA::MPI::BlockVector       system_normal_RHS;         // system right-hand side for computing the normal vector
    LA::MPI::Vector            curvature_field;
    LA::MPI::Vector            system_curvature_RHS;         // system right-hand side for computing the normal vector
    //LA::MPI::BlockVector       advection_field;         // system right-hand side for computing the normal vector

    std::vector<double>        volume_fraction;
    IndexSet                   locally_owned_dofs;
    IndexSet                   locally_relevant_dofs;
    ConditionalOStream         pcout;
    bool                       normalsComputed;
    TimerOutput                computing_timer;
    Timer                      timer;
    TensorFunction<1, dim> &   AdvectionField;

    Reinitialization<dim>      reini;
  };
} // end of namespace LevelSet
