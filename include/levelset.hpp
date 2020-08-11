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
  class LevelSetEquation
  {
  public:
    LevelSetEquation(const LevelSetParameters& parameters,
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


  template <int dim>
  LevelSetEquation<dim>::LevelSetEquation(
                     const LevelSetParameters& parameters_,
                     Triangulation<dim>&       triangulation_)
    : epsilon ( GridTools::minimal_cell_diameter(triangulation_) * 2.0 )
    , parameters(       parameters_)
    , fe(               parameters_.levelset_degree )
    , triangulation(    triangulation_ )
    , dof_handler(      triangulation_ )
    , qGauss(           QGauss<dim>(parameters_.levelset_degree+1) )
    , time_step(        parameters_.time_step_size )
    , time(             time_step )
    , timestep_number(  1 )
  {}
  
  template <int dim>
  void LevelSetEquation<dim>::setInitialConditions(const Function<dim>& InitialValues)
  {
    VectorTools::project(dof_handler, 
                         constraints,
                         qGauss,
                         InitialValues,           
                         solution_u);
  } 
  
  template <int dim>
  void LevelSetEquation<dim>::setDirichletBoundaryConditions(const Function<dim>& DirichletValues)
  {
    std::map<types::global_dof_index, double> boundary_values;
    
    VectorTools::interpolate_boundary_values( dof_handler,
                                              utilityFunctions::BoundaryConditions::Types::dirichlet,
                                              DirichletValues,
                                              boundary_values );

    MatrixTools::apply_boundary_values(boundary_values,
                                       systemMatrix,
                                       solution_u,
                                       systemRHS);
  } 
  
  template <int dim>
  void LevelSetEquation<dim>::assembleSystemMatrices(TensorFunction<1, dim> &AdvectionField_)
  {
    systemMatrix = 0.0;
    systemRHS = 0.0;
    
    FEValues<dim> fe_values( fe,
                             qGauss,
                             update_values | update_gradients | update_JxW_values | update_quadrature_points );

    const unsigned int dofs_per_cell =   fe.dofs_per_cell;
    const unsigned int n_q_points    =   qGauss.size();

    std::vector<types::global_dof_index> globalDofIndices( dofs_per_cell );
    
    std::vector<double>         phiAtQ(  n_q_points );
    std::vector<Tensor<1,dim>>  phiGradAtQ( n_q_points, Tensor<1,dim>() );

    FullMatrix<double> cellMatrix; 
    cellMatrix.reinit(dofs_per_cell, dofs_per_cell);
    
    Vector<double> cellRHS; 
    cellRHS.reinit( dofs_per_cell );
    AdvectionField_.set_time(time);
    
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        cellMatrix = 0;
        cellRHS = 0;
        fe_values.reinit(cell);
        fe_values.get_function_values(     solution_u, phiAtQ ); // compute values of old solution
        fe_values.get_function_gradients(  solution_u, phiGradAtQ ); // compute values of old solution

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
            auto qCoord = fe_values.get_quadrature_points()[q_index];
            Tensor<1, dim> a = AdvectionField_.value( qCoord );

            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                auto velocityGradShape = a * fe_values.shape_grad( i, q_index);  // grad_phi_j(x_q)

                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                    cellMatrix( i, j ) += (  fe_values.shape_value( i, q_index) * 
                                             fe_values.shape_value( j, q_index) +
                                             parameters.theta * time_step * ( parameters.artificial_diffusivity * 
                                                                   fe_values.shape_grad( i, q_index) * 
                                                                   fe_values.shape_grad( j, q_index) -
                                                                   velocityGradShape * 
                                                                   fe_values.shape_value( j, q_index) )
                                          ) * fe_values.JxW(q_index);                                    
                    
                }

                cellRHS( i ) +=
                   (  fe_values.shape_value( i, q_index) * phiAtQ[q_index]
                       - 
                      ( 1. - parameters.theta ) * time_step * (
                              parameters.artificial_diffusivity *
                              fe_values.shape_grad( i, q_index) *
                              phiGradAtQ[q_index]
                              -
                              a * fe_values.shape_grad(  i, q_index)
                              * phiAtQ[q_index]    
                         )
                   ) * fe_values.JxW(q_index) ;       // dx
            }
          }// end gauss
          
        
        // assembly
        cell->get_dof_indices(globalDofIndices);

        for (const unsigned int i : fe_values.dof_indices())
          for (const unsigned int j : fe_values.dof_indices())
          {
            systemMatrix.add( globalDofIndices[i],
                                   globalDofIndices[j],
                                      cellMatrix(i, j));
          }

        for (const unsigned int i : fe_values.dof_indices())
            systemRHS( globalDofIndices[i] ) += cellRHS(i);
         
      }

  }
  
  template <int dim>
  void LevelSetEquation<dim>::computeReinitializationMatrices( const double dTau )
  {
    systemRHS = 0;
    systemMatrix = 0.0;
    
    FEValues<dim> fe_values( fe,
                             qGauss,
                             update_values | update_gradients | update_quadrature_points | update_JxW_values );

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = qGauss.size();
    
    FullMatrix<double>   cell_matrix( dofs_per_cell, dofs_per_cell );
    Vector<double>       cell_rhs(    dofs_per_cell );
    
    std::vector<double>         psiAtQ(     n_q_points );
    std::vector<Tensor<1,dim>>  normalAtQ(  n_q_points, Tensor<1,dim>() );
    std::vector<Tensor<1,dim>>  psiGradAtQ( n_q_points, Tensor<1,dim>() );
    
    std::vector<types::global_dof_index> globalDofIndices( dofs_per_cell );
    
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        
        cell_matrix = 0.0;
        cell_rhs = 0.0;
        
        cell->get_dof_indices(             globalDofIndices);
        fe_values.get_function_values(     re_solution_u, psiAtQ ); // compute values of old solution
        fe_values.get_function_gradients(  re_solution_u, psiGradAtQ ); // compute values of old solution
        fe_values.get_function_gradients(  solution_u, normalAtQ ); // compute normals from level set solution at tau=0
        
        for (auto& n : normalAtQ)
            n /= n.norm();
        
        // @todo: only compute normals once during timestepping
        for (const unsigned int q_index : fe_values.quadrature_point_indices())
         {
            const double diffRhs = epsilon * normalAtQ[q_index] * psiGradAtQ[q_index];

            for (const unsigned int i : fe_values.dof_indices())
            {
                const double nTimesGradient_i = normalAtQ[q_index] * fe_values.shape_grad(i, q_index);

                for (const unsigned int j : fe_values.dof_indices())
                {
                    const double nTimesGradient_j = normalAtQ[q_index] * fe_values.shape_grad(j, q_index);
                    cell_matrix(i,j) += (
                                          fe_values.shape_value(i,q_index) * fe_values.shape_value(j,q_index)
                                          + 
                                          dTau * epsilon * nTimesGradient_i * nTimesGradient_j
                                        ) 
                                        * fe_values.JxW( q_index );
                }

                cell_rhs(i) += ( 0.5 * ( 1. - psiAtQ[q_index] * psiAtQ[q_index] ) - diffRhs )
                                *
                                nTimesGradient_i 
                                *
                                dTau 
                                * 
                                fe_values.JxW( q_index );
                
            }                                    
        }// end loop over gauss points
        
      // assembly
      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          systemMatrix.add( globalDofIndices[i],
                                 globalDofIndices[j],
                                 cell_matrix(i, j));

      for (const unsigned int i : fe_values.dof_indices())
        systemRHS[ globalDofIndices[i] ] += cell_rhs(i);
     
    }
  }

  template <int dim>
  void LevelSetEquation<dim>::computeDampedNormalLevelSet()
  {
    systemMatrix = 0.0;
    for (unsigned int d=0; d<dim; ++d)
        system_normal_RHS.block(d) = 0.0;
    
    FEValues<dim> fe_values( fe,
                             qGauss,
                             update_values | update_gradients | update_quadrature_points | update_JxW_values );
    const unsigned int n_q_points    = qGauss.size();

    const unsigned int          dofs_per_cell = fe.dofs_per_cell;
    FullMatrix<double>          normal_cell_matrix( dofs_per_cell, dofs_per_cell );
    std::vector<Vector<double>> normal_cell_rhs(dim, Vector<double>(dofs_per_cell) );
    
    std::vector<types::global_dof_index> local_dof_indices( dofs_per_cell );
    std::vector<Tensor<1,dim>>           normal_at_q(  n_q_points, Tensor<1,dim>() );

    const double damping = GridTools::minimal_cell_diameter(triangulation) * 0.5; //@todo: modifiy damping parameter

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        cell->get_dof_indices( local_dof_indices );
        
        normal_cell_matrix = 0.0;
        for(auto& normal_cell : normal_cell_rhs)
            normal_cell =    0.0;

        fe_values.get_function_gradients(  solution_u, normal_at_q ); // compute normals from level set solution at tau=0
        
        for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                const double phi_i             = fe_values.shape_value(i, q_index);
                const Tensor<1,dim> grad_phi_i = fe_values.shape_grad(i, q_index);
                
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                    const double phi_j             = fe_values.shape_value(j, q_index);
                    const Tensor<1,dim> grad_phi_j = fe_values.shape_grad(j, q_index);

                    normal_cell_matrix( i, j ) += ( 
                                                phi_i * phi_j 
                                                + 
                                                damping * grad_phi_i * grad_phi_j  
                                                )
                                                * 
                                                fe_values.JxW( q_index ) ;
                }
 
                for (unsigned int d=0; d<dim; ++d)
                {
                    normal_cell_rhs[d](i) +=   phi_i
                                               * 
                                               normal_at_q[ q_index ][ d ]  
                                               * 
                                               fe_values.JxW( q_index );
                }
            }
        }
        
        // assembly
        for (const unsigned int i : fe_values.dof_indices())
          for (const unsigned int j : fe_values.dof_indices())
            systemMatrix.add( local_dof_indices[i],
                              local_dof_indices[j],
                              normal_cell_matrix(i, j));

        for (unsigned int d=0; d<dim; ++d)
            for (const unsigned int i : fe_values.dof_indices())
                system_normal_RHS.block(d)[ local_dof_indices[ i ] ] += normal_cell_rhs[ d ](i);
    } // end for loop over cells

    for (unsigned int d=0; d<dim; ++d)
        solve_cg(system_normal_RHS.block( d ), systemMatrix, solution_normal_damped.block( d ), "damped normals");
  }

  template <int dim>
  void LevelSetEquation<dim>::computeDampedCurvatureLevelSet()
  {
    systemMatrix = 0.0; // @currently system matrix from computeDampedNormal is reused
    system_curvature_RHS = 0.0;
    
    //TimerOutput::Scope timer (*this->timer, "Curvature computation.");

    FEValues<dim> fe_values( fe,
                             qGauss,
                             update_values | update_gradients | update_quadrature_points | update_JxW_values );

    const unsigned int n_q_points    = qGauss.size();
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    
    std::vector<types::global_dof_index> local_dof_indices(     dofs_per_cell );
    FullMatrix<double>                   curvature_cell_matrix( dofs_per_cell, dofs_per_cell );
    Vector<double>                       curvature_cell_rhs(    dofs_per_cell );
    std::vector<Tensor<1,dim>>           normal_at_q(           n_q_points, Tensor<1,dim>() );

    const double damping = 0.0; //@todo: modifiy damping parameter

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit( cell );
        
        curvature_cell_matrix = 0.0;
        curvature_cell_rhs    = 0.0;
 
        for (unsigned int d=0; d<dim; ++d)
        {
            std::vector<double> temp(n_q_points);
            fe_values.get_function_values( solution_normal_damped.block(d), temp ); 
            for (const unsigned int q_index : fe_values.quadrature_point_indices())
                normal_at_q[q_index][d] = temp[q_index];
        }

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
            normal_at_q[q_index] /= normal_at_q[q_index].norm();
            

            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                const double phi_i             = fe_values.shape_value( i, q_index );
                const Tensor<1,dim> grad_phi_i = fe_values.shape_grad(  i, q_index );
                
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                    const double phi_j             = fe_values.shape_value( j, q_index);
                    const Tensor<1,dim> grad_phi_j = fe_values.shape_grad(  j, q_index);

                    curvature_cell_matrix( i, j ) += ( phi_i * phi_j 
                                                       + 
                                                       damping * grad_phi_i * grad_phi_j  
                                                     )
                                                       * 
                                                       fe_values.JxW( q_index ) ;
                }
                curvature_cell_rhs(i) += ( grad_phi_i
                                           * 
                                           normal_at_q[ q_index ] 
                                           * 
                                           fe_values.JxW( q_index ) );
            }
        }
        
        cell->get_dof_indices( local_dof_indices );
        
        // assembly
        for (const unsigned int i : fe_values.dof_indices())
          for (const unsigned int j : fe_values.dof_indices())
            systemMatrix.add( local_dof_indices[i],
                              local_dof_indices[j],
                              curvature_cell_matrix(i, j));

        for (const unsigned int i : fe_values.dof_indices())
            system_curvature_RHS[ local_dof_indices[ i ] ] += curvature_cell_rhs(i);

    } // end for loop over cells
  
    solve_cg(system_curvature_RHS, systemMatrix, solution_curvature_damped, "damped curvature");
  }

  template <int dim>
  void LevelSetEquation<dim>::solve_cg(const Vector<double>& RHS,
                                       const SparseMatrix<double>& matrix,
                                       Vector<double>& solution,
                                       const std::string& callerFunction)
  {
    SolverControl                   solver_control( 1000, 1e-8 * RHS.l2_norm() );
    SolverCG<Vector<double>>        cg( solver_control );
    cg.solve( matrix, 
              solution, 
              RHS, 
              PreconditionIdentity() );
    std::cout << "cg solver called by " << callerFunction << " with "  << solver_control.last_step() << " CG iterations." << std::endl;
  }

  template <int dim>
  void LevelSetEquation<dim>::setup_system()
  {
    dof_handler.distribute_dofs( fe );

    std::cout << "Number of active cells: "       << triangulation.n_active_cells() << std::endl;
    std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()           << std::endl << std::endl;

    DynamicSparsityPattern dsp(      dof_handler.n_dofs(), dof_handler.n_dofs() );
    DoFTools::make_sparsity_pattern( dof_handler, dsp );
    sparsity_pattern.copy_from(      dsp );
    
    systemMatrix.reinit(             sparsity_pattern );
                                      
    systemRHS.reinit(                dof_handler.n_dofs() );

    system_normal_RHS.reinit( dim,   dof_handler.n_dofs() );
    system_normal_RHS.collect_sizes(); 
    
    solution_normal_damped.reinit( dim, dof_handler.n_dofs() );
    //for (unsigned int d=0; d<dim; ++d)
        //solution_normal_damped.block(d).reinit(dof_handler.n_dofs());

    solution_normal_damped.collect_sizes(); 
    
    system_curvature_RHS.reinit(      dof_handler.n_dofs() );
    solution_curvature_damped.reinit( dof_handler.n_dofs() );
    
    solution_u.reinit(               dof_handler.n_dofs() );
    
    re_solution_u.reinit(            dof_handler.n_dofs() );
    re_delta_solution_u.reinit(      dof_handler.n_dofs() );
    
    advection_x.reinit(              dof_handler.n_dofs() ); //todo
    advection_y.reinit(              dof_handler.n_dofs() );

    constraints.close();
  }

  template <int dim>
  void LevelSetEquation<dim>::solve_u()
  {
    SolverControl            solver_control( 1000, 1e-8 * systemRHS.l2_norm() );
    SolverGMRES<Vector<double>>       gmres( solver_control );

    gmres.solve( systemMatrix, 
                 solution_u, 
                 systemRHS, 
                 PreconditionIdentity() );

    std::cout << "   u-equation: " << solver_control.last_step() << " GMRES iterations." << std::endl;
  }
  
  //@ todo: merge two solve functions using call by reference
  template <int dim>
  void LevelSetEquation<dim>::re_solve_u()
  {
    SolverControl            solver_control( 1000, 1e-8 * systemRHS.l2_norm() );
    SolverGMRES<Vector<double>>       gmres( solver_control );

    gmres.solve( systemMatrix, 
                 re_delta_solution_u, 
                 systemRHS, 
                 PreconditionIdentity() );

    re_solution_u += re_delta_solution_u;
  }
  
  template <int dim>
  void LevelSetEquation<dim>::computeAdvection(TensorFunction<1, dim> &AdvectionField_)
  {
    std::map<types::global_dof_index, Point<dim> > supportPoints;
    DoFTools::map_dofs_to_support_points<dim,dim>(MappingQGeneric<dim>(fe.degree),dof_handler,supportPoints);

    for(auto& globalDofAndNodeCoord : supportPoints)
    {
        auto a = AdvectionField_.value(globalDofAndNodeCoord.second);
        advection_x[globalDofAndNodeCoord.first] = a[0];
        advection_y[globalDofAndNodeCoord.first] = a[1];
    } 
  }

  // @ to be rearranged
  template <int dim>
  void LevelSetEquation<dim>::computeVolume( )
  {
    FEValues<dim> fe_values( fe,
                             qGauss,
                             update_values | update_gradients | update_JxW_values | update_quadrature_points );

    const unsigned int dofs_per_cell =   fe.dofs_per_cell;
    std::vector<types::global_dof_index> globalDofIndices( dofs_per_cell );
    
    double volumeOfPhase1 = 0;
    double volumeOfPhase2 = 0;
    const unsigned int n_q_points    = qGauss.size();
    std::vector<double> phiAtQ(  n_q_points );
    
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(               cell );
        cell->get_dof_indices(          globalDofIndices );
        fe_values.get_function_values(  solution_u, phiAtQ ); // compute values of old solution

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
            volumeOfPhase1 += ( phiAtQ[q_index] * 0.5 + 0.5 ) * fe_values.JxW(q_index); 
            volumeOfPhase2 += ( 1. - ( phiAtQ[q_index] * 0.5 + 0.5 ) ) * fe_values.JxW(q_index);
        }
    }
    volumeOfPhase1PerTimeStep.push_back(volumeOfPhase1);
    volumeOfPhase2PerTimeStep.push_back(volumeOfPhase2);
  }

  template <int dim>
  void LevelSetEquation<dim>::output_results(const double timeStep)
  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution_u,                      "phi");
    data_out.add_data_vector(advection_x ,                    "a_x");
    data_out.add_data_vector(advection_y,                     "a_y");
    data_out.add_data_vector(solution_normal_damped.block(0), "normal_x");
    data_out.add_data_vector(solution_normal_damped.block(1), "normal_y");
    data_out.add_data_vector(solution_curvature_damped,       "curvature");
    data_out.build_patches();

    const std::string filename = "solution-" + Utilities::int_to_string(timeStep, 3) + ".vtu";
    DataOutBase::VtkFlags vtk_flags;
    vtk_flags.compression_level = DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
    data_out.set_flags(vtk_flags);
    std::ofstream output(filename);
    data_out.write_vtu(output);
    
  }
  
  template <int dim>
  void LevelSetEquation<dim>::run( const Function<dim>& InitialValues,
                                   TensorFunction<1, dim>& AdvectionField_,
                                   const Function<dim>& DirichletValues) 
  {
    std::cout << "setup system " << std::endl;
    setup_system();
    std::cout << "initial conditions set " << std::endl;
    setInitialConditions(InitialValues);
    timeVector.push_back(0.0);

    timestep_number=0;
    computeAdvection(AdvectionField_);
    //computeDampedNormalLevelSet();
    //computeDampedCurvatureLevelSet();

    output_results( timestep_number );    // print initial state
    computeVolume( );

    timestep_number++; 
    for (; time <= parameters.end_time; time += parameters.time_step_size, ++timestep_number)
      {
        timeVector.push_back(time);
        std::cout << "Time step " << timestep_number << " at t=" << time << std::endl;

        assembleSystemMatrices(AdvectionField_); // @todo: insert updateFlag
        setDirichletBoundaryConditions(DirichletValues); 
        
        solve_u();
        
        if ( parameters.activate_reinitialization )    
        {
            re_solution_u     = solution_u;
            const double re_time_step       = 1./(dim*dim) * GridTools::minimal_cell_diameter(triangulation); 
            unsigned int re_timestep_number = 0;
            double re_time                  = 0.0;
            
            std::cout << "       >>>>>>>>>>>>>>>>>>> REINITIALIZATION START " << std::endl;
            for (; re_timestep_number < parameters.max_n_reinit_steps; re_time += re_time_step, ++re_timestep_number) // 3 to 5 timesteps are enough to reach steady state according to Kronbichler et al.
            {
                computeReinitializationMatrices(re_time_step);
                re_solve_u();
                
                std::cout << "      | Time step " << re_timestep_number << " at tau=" << std::fixed << std::setprecision(5) << re_time << "\t |R|∞ = " << re_delta_solution_u.linfty_norm() << std::endl;
                
                if (re_delta_solution_u.linfty_norm() < 1e-6)
                    break;
                
            }
            std::cout << "       <<<<<<<<<<<<<<<<<<< REINITIALIZATION END " << std::endl;

            solution_u     = re_solution_u;
        }

        computeAdvection(AdvectionField_);
        //computeDampedNormalLevelSet();
        //computeDampedCurvatureLevelSet();
        output_results(timestep_number);
        
        if (parameters.compute_volume)
        {
            std::cout << " n = " << timestep_number << std::endl;
            computeVolume();
            std::cout << "curr volume -- phase 1 " << volumeOfPhase1PerTimeStep[timestep_number] << std::endl;
            std::cout << "curr volume -- phase 2 " << volumeOfPhase2PerTimeStep[timestep_number] << std::endl;

        }   
    }
  }
} // end of namespace LevelSet
