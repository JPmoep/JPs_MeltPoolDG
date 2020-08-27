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
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>


#include <deal.II/fe/mapping.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
//#include <deal.II/lac/solver_cg.h> // only for symmetric matrices
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
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
#include <iomanip>
#include <algorithm> // for enum ::All
//#include <fmt/format.h>  // unfortunately does not work currently
#include <cmath>
#include "utilityFunctions.hpp"

namespace LevelSet
{
  using namespace dealii; 
  
  template <int dim>
  class LevelSetEquationNotParallel
  {
  public:
    LevelSetEquationNotParallel(const LevelSetParameters& parameters,
                     Triangulation<dim>&       triangulation);
    void run( const Function<dim>& InitialValues,
              TensorFunction<1, dim> &AdvectionField_,
              const Function<dim>& DirichletValues);
    double epsilon;

  private:
    void setup_system();
    void setInitialConditions(const Function<dim>& InitialValues);
    void setDirichletBoundaryConditions(const Function<dim>& DirichletValues);
    void assembleSystemMatrices(TensorFunction<1, dim> &AdvectionField_);
    void computeReinitializationMatrices( const double dTau );
    void solve_u();
    void re_solve_u();
    void solve_cg( const Vector<double>& RHS, 
                   const SparseMatrix<double>& matrix,
                   Vector<double>& solution,
                   const std::string& callerFunction = " "
                   );
    void output_results(const double timeStep);
    void computeVolume();
    void computeAdvection(TensorFunction<1, dim> &AdvectionField_);
    void computeDampedNormalLevelSet();
    void computeDampedCurvatureLevelSet();
    
    LevelSetParameters        parameters;
    FE_Q<dim>                 fe;
    Triangulation<dim>&       triangulation;
    DoFHandler<dim>           dof_handler;
    QGauss<dim>               qGauss; 
    double                    time_step;
    double                    time;
    unsigned int              timestep_number;
    
    AffineConstraints<double> constraints;

    SparsityPattern           sparsity_pattern;

    SparseMatrix<double>      systemMatrix;              // global system matrix
    Vector<double>            systemRHS;                 // global system right-hand side
    BlockVector<double>       system_normal_RHS;         // system right-hand side for computing the normal vector
    Vector<double>            system_curvature_RHS;      // system right-hand side for computing the curvature
    BlockVector<double>       solution_normal_damped;    // solution of the (damped) normal vector
    Vector<double>            solution_curvature_damped; // solution of the (damped) curvature
    
    Vector<double>            old_solution_u;
    Vector<double>            solution_u;
    Vector<double>            re_solution_u;
    Vector<double>            re_delta_solution_u;

    Vector<double>            advection_x;
    Vector<double>            advection_y;
    
    std::vector<double>       volumeOfPhase1PerTimeStep;
    std::vector<double>       volumeOfPhase2PerTimeStep;
    std::vector<double>       timeVector;
  };
} // end of namespace LevelSet
