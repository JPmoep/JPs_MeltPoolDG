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
#include <iomanip>
#include <algorithm> // for enum ::All
//#include <fmt/format.h>  // unfortunately does not work currently
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


namespace LevelSetParallel
{
  using namespace dealii; 
  
  template <int dim>
  class LevelSetEquation
  {
  public:
    LevelSetEquation(const LevelSetParameters& parameters,
                     parallel::distributed::Triangulation<dim>&       triangulation,
                     MPI_Comm& mpi_commun);
    void run( const Function<dim>& InitialValues,
              TensorFunction<1, dim> &AdvectionField_,
              const Function<dim>& DirichletValues);
    double epsilon;

  private:
    void setup_system(const Function<dim>& DirichletValues );
    void setInitialConditions(const Function<dim>& InitialValues);
    void assembleSystemMatrices( TensorFunction<1, dim> &AdvectionField_,
                                 const Function<dim>& DirichletValues );
    void computeReinitializationMatrices( const double dTau );
    void solve_u();
    void re_solve_u();
    void solve_cg(const LA::MPI::Vector& RHS,
                                       const LA::MPI::SparseMatrix& matrix,
                                       LA::MPI::Vector& solution,
                                       const std::string& callerFunction);
    //void solve_cg( const Vector<double>& RHS, 
                   //const SparseMatrix<double>& matrix,
                   //Vector<double>& solution,
                   //const std::string& callerFunction = " "
                   //);
    void output_results(const double timeStep);
    //void computeVolume();
    //void computeAdvection(TensorFunction<1, dim> &AdvectionField_);
    void computeDampedNormalLevelSet();
    //void computeDampedCurvatureLevelSet();
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
    //Vector<double>            system_curvature_RHS;      // system right-hand side for computing the curvature
    //BlockVector<double>       solution_normal_damped;    // solution of the (damped) normal vector
    //Vector<double>            solution_curvature_damped; // solution of the (damped) curvature
    
    //Vector<double>            old_solution_u;
    LA::MPI::Vector             re_solution_u;
    LA::MPI::Vector             re_delta_solution_u;
    LA::MPI::BlockVector        normal_vector_field;
    LA::MPI::BlockVector        system_normal_RHS;         // system right-hand side for computing the normal vector

    //Vector<double>            advection_x;
    //Vector<double>            advection_y;
    
    //std::vector<double>       volumeOfPhase1PerTimeStep;
    //std::vector<double>       volumeOfPhase2PerTimeStep;
    //std::vector<double>       timeVector;
    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;
    ConditionalOStream pcout;
    bool normalsComputed;
  };


  template <int dim>
  LevelSetEquation<dim>::LevelSetEquation(
                     const LevelSetParameters& parameters_,
                     parallel::distributed::Triangulation<dim>&       triangulation_,
                     MPI_Comm& mpi_commun)
    : mpi_communicator( mpi_commun)
    , epsilon ( GridTools::minimal_cell_diameter(triangulation_) * 2.0 )
    , parameters(       parameters_)
    , fe(               parameters_.levelset_degree )
    , triangulation(    triangulation_ )
    , dof_handler(      triangulation_ )
    , qGauss(           QGauss<dim>(parameters_.levelset_degree+1) )
    , time_step(        parameters_.time_step_size )
    , time(             time_step )
    , timestep_number(  1 )
    , pcout(std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  {}
  
  
  template <int dim>
  void LevelSetEquation<dim>::setup_system(const Function<dim>& DirichletValues )
  {
    dof_handler.distribute_dofs( fe );

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    
    pcout << "Number of active cells: "       << triangulation.n_active_cells() << std::endl;
    pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()           << std::endl << std::endl;
                                      
    systemRHS.reinit(locally_owned_dofs, mpi_communicator); 
    
    solution_u.reinit(locally_owned_dofs,
                      locally_relevant_dofs,
                       mpi_communicator);
    
    re_solution_u.reinit(locally_owned_dofs,
                      locally_relevant_dofs,
                       mpi_communicator);
    
    re_delta_solution_u.reinit(locally_owned_dofs,
                       mpi_communicator);
    
    system_normal_RHS.reinit( dim ); 
    normal_vector_field.reinit( dim); 
    
    for (unsigned int d=0; d<dim; ++d)
    {
        system_normal_RHS.block(d).reinit(locally_owned_dofs, mpi_communicator); 
        
        normal_vector_field.block(d).reinit(locally_owned_dofs,
                                   locally_relevant_dofs,
                                    mpi_communicator);
    }   
    system_normal_RHS.collect_sizes(); 
    normal_vector_field.collect_sizes(); 

    // constraints for level set function
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints); 
    VectorTools::interpolate_boundary_values( dof_handler,
                                              utilityFunctions::BoundaryConditions::Types::dirichlet,
                                              DirichletValues,
                                              constraints );
    constraints.close();   

    DynamicSparsityPattern dsp( locally_relevant_dofs );
    DoFTools::make_sparsity_pattern( dof_handler, dsp, constraints, false );
    SparsityTools::distribute_sparsity_pattern(dsp,
                                           dof_handler.locally_owned_dofs(),
                                           mpi_communicator,
                                           locally_relevant_dofs);

    systemMatrix.reinit( locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);
    
    // constraints for reinitialization function
    constraints_re.clear();
    constraints_re.reinit(locally_relevant_dofs);
    //DoFTools::make_hanging_node_constraints(dof_handler, constraints_re); 
    constraints_re.close();   
    
    DynamicSparsityPattern dsp_re( locally_relevant_dofs );
    DoFTools::make_sparsity_pattern( dof_handler, dsp_re, constraints_re, false );
    SparsityTools::distribute_sparsity_pattern(dsp_re,
                                               dof_handler.locally_owned_dofs(),
                                               mpi_communicator,
                                               locally_relevant_dofs);
    
    systemMatrix_re.reinit( locally_owned_dofs,
                            locally_owned_dofs,
                            dsp_re,
                            mpi_communicator );
    
    
    //system_normal_RHS.reinit( dim,   dof_handler.n_dofs() );
    //system_normal_RHS.collect_sizes(); 
    
    //solution_normal_damped.reinit( dim, dof_handler.n_dofs() );
    ////for (unsigned int d=0; d<dim; ++d)
        ////solution_normal_damped.block(d).reinit(dof_handler.n_dofs());

    //solution_normal_damped.collect_sizes(); 
    
    //system_curvature_RHS.reinit(      dof_handler.n_dofs() );
    //solution_curvature_damped.reinit( dof_handler.n_dofs() );
    
    
    //re_solution_u.reinit(            dof_handler.n_dofs() );
    //re_delta_solution_u.reinit(      dof_handler.n_dofs() );
    
    //advection_x.reinit(              dof_handler.n_dofs() ); //todo
    //advection_y.reinit(              dof_handler.n_dofs() );

  }
  
  template <int dim>
  void LevelSetEquation<dim>::setInitialConditions(const Function<dim>& InitialValues)
  {
    LA::MPI::Vector solutionTemp( dof_handler.locally_owned_dofs(), mpi_communicator);

    VectorTools::project(dof_handler, 
                         constraints,
                         qGauss,
                         InitialValues,           
                         solutionTemp);

    solution_u = solutionTemp;
  } 
  
  template <int dim>
  void LevelSetEquation<dim>::assembleSystemMatrices(TensorFunction<1, dim> &AdvectionField_,
                                                    const Function<dim>& DirichletValues)
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

    FullMatrix<double> cellMatrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cellRHS( dofs_per_cell );
    
    AdvectionField_.set_time(time);
    
    for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
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
        constraints.distribute_local_to_global(cellMatrix,
                                               cellRHS,
                                               globalDofIndices,
                                               systemMatrix,
                                               systemRHS);
         
      }
      systemMatrix.compress(VectorOperation::add);
      systemRHS.compress(VectorOperation::add);
  }
  
  template <int dim>
  void LevelSetEquation<dim>::computeReinitializationMatrices( const double dTau )
  {
    systemRHS = 0;
    systemMatrix_re = 0.0;
    
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
    
    bool normalsQuick = false;

    for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
    {
        fe_values.reinit(cell);
        
        cell_matrix = 0.0;
        cell_rhs = 0.0;
        
        fe_values.get_function_values(     re_solution_u, psiAtQ ); // compute values of old solution
        fe_values.get_function_gradients(  re_solution_u, psiGradAtQ ); // compute values of old solution

        if (normalsQuick)
        {
            fe_values.get_function_gradients( solution_u, normalAtQ ); // compute normals from level set solution at tau=0
            for (auto& n : normalAtQ)
                n /= n.norm(); //@todo: add exception
        }
        else
        {
            for (unsigned int d=0; d<dim; ++d )
            {
                std::vector<double> temp (n_q_points);
                fe_values.get_function_values(  normal_vector_field.block(d), temp); // compute normals from level set solution at tau=0
                for (const unsigned int q_index : fe_values.quadrature_point_indices())
                {
                    normalAtQ[q_index][d] = temp[q_index];
                }
            }
            for (auto& n : normalAtQ)
                n /= n.norm(); //@todo: add exception
        }

        // @todo: only compute normals once during timestepping
        for (const unsigned int q_index : fe_values.quadrature_point_indices())
         {
            const double diffRhs = epsilon * normalAtQ[q_index] * psiGradAtQ[q_index];

            for (const unsigned int i : fe_values.dof_indices())
            {
                //if (!normalsComputed)
                //{
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
                //}

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
        cell->get_dof_indices(globalDofIndices);
        constraints_re.distribute_local_to_global(cell_matrix,
                                                  cell_rhs,
                                                  globalDofIndices,
                                                  systemMatrix_re,
                                                  systemRHS);
         
      }
      systemMatrix_re.compress(VectorOperation::add);
      systemRHS.compress(VectorOperation::add);
      normalsComputed = true;  
  }

  template <int dim>
  void LevelSetEquation<dim>::computeDampedNormalLevelSet()
  {
    systemMatrix_re = 0.0;
    for (unsigned int d=0; d<dim; d++)
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
    if (cell->is_locally_owned())
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
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int d=0; d<dim; ++d)
            constraints_re.distribute_local_to_global(normal_cell_matrix,
                                                      normal_cell_rhs[d],
                                                      local_dof_indices,
                                                      systemMatrix_re,
                                                      system_normal_RHS.block(d));
         
      }
      systemMatrix_re.compress(VectorOperation::add);
      for (unsigned int d=0; d<dim; ++d)
        system_normal_RHS.block(d).compress(VectorOperation::add);
      normalsComputed = true;  

    for (unsigned int d=0; d<dim; ++d)
        solve_cg(system_normal_RHS.block( d ), systemMatrix_re, normal_vector_field.block( d ), "damped normals");
  
  
  }

  //template <int dim>
  //void LevelSetEquation<dim>::computeDampedCurvatureLevelSet()
  //{
    //systemMatrix = 0.0; // @currently system matrix from computeDampedNormal is reused
    //system_curvature_RHS = 0.0;
    
    ////TimerOutput::Scope timer (*this->timer, "Curvature computation.");

    //FEValues<dim> fe_values( fe,
                             //qGauss,
                             //update_values | update_gradients | update_quadrature_points | update_JxW_values );

    //const unsigned int n_q_points    = qGauss.size();
    //const unsigned int dofs_per_cell = fe.dofs_per_cell;
    
    //std::vector<types::global_dof_index> local_dof_indices(     dofs_per_cell );
    //FullMatrix<double>                   curvature_cell_matrix( dofs_per_cell, dofs_per_cell );
    //Vector<double>                       curvature_cell_rhs(    dofs_per_cell );
    //std::vector<Tensor<1,dim>>           normal_at_q(           n_q_points, Tensor<1,dim>() );

    //const double damping = 0.0; //@todo: modifiy damping parameter

    //for (const auto &cell : dof_handler.active_cell_iterators())
    //{
        //fe_values.reinit( cell );
        
        //curvature_cell_matrix = 0.0;
        //curvature_cell_rhs    = 0.0;
 
        //for (unsigned int d=0; d<dim; ++d)
        //{
            //std::vector<double> temp(n_q_points);
            //fe_values.get_function_values( solution_normal_damped.block(d), temp ); 
            //for (const unsigned int q_index : fe_values.quadrature_point_indices())
                //normal_at_q[q_index][d] = temp[q_index];
        //}

        //for (const unsigned int q_index : fe_values.quadrature_point_indices())
        //{
            //normal_at_q[q_index] /= normal_at_q[q_index].norm();
            

            //for (unsigned int i=0; i<dofs_per_cell; ++i)
            //{
                //const double phi_i             = fe_values.shape_value( i, q_index );
                //const Tensor<1,dim> grad_phi_i = fe_values.shape_grad(  i, q_index );
                
                //for (unsigned int j=0; j<dofs_per_cell; ++j)
                //{
                    //const double phi_j             = fe_values.shape_value( j, q_index);
                    //const Tensor<1,dim> grad_phi_j = fe_values.shape_grad(  j, q_index);

                    //curvature_cell_matrix( i, j ) += ( phi_i * phi_j 
                                                       //+ 
                                                       //damping * grad_phi_i * grad_phi_j  
                                                     //)
                                                       //* 
                                                       //fe_values.JxW( q_index ) ;
                //}
                //curvature_cell_rhs(i) += ( grad_phi_i
                                           //* 
                                           //normal_at_q[ q_index ] 
                                           //* 
                                           //fe_values.JxW( q_index ) );
            //}
        //}
        
        //cell->get_dof_indices( local_dof_indices );
        
        //// assembly
        //for (const unsigned int i : fe_values.dof_indices())
          //for (const unsigned int j : fe_values.dof_indices())
            //systemMatrix.add( local_dof_indices[i],
                              //local_dof_indices[j],
                              //curvature_cell_matrix(i, j));

        //for (const unsigned int i : fe_values.dof_indices())
            //system_curvature_RHS[ local_dof_indices[ i ] ] += curvature_cell_rhs(i);

    //} // end for loop over cells
  
    //solve_cg(system_curvature_RHS, systemMatrix, solution_curvature_damped, "damped curvature");
  //}

  template <int dim>
  void LevelSetEquation<dim>::solve_cg(const LA::MPI::Vector& RHS,
                                       const LA::MPI::SparseMatrix& matrix,
                                       LA::MPI::Vector& solution,
                                       const std::string& callerFunction)
  {
    LA::MPI::Vector    completely_distributed_solution(locally_owned_dofs,
                                                       mpi_communicator);
    SolverControl            solver_control( dof_handler.n_dofs(), 1e-8 * systemRHS.l2_norm() );
#ifdef USE_PETSC_LA
    LA::SolverCG solver(solver_control, mpi_communicator);
#else
    LA::SolverCG solver(solver_control);
#endif

    LA::MPI::PreconditionAMG preconditioner;
    LA::MPI::PreconditionAMG::AdditionalData data;
    preconditioner.initialize(matrix, data);

    solver.solve( matrix, 
                  completely_distributed_solution, 
                  RHS, 
                  preconditioner );
    solution = completely_distributed_solution;
    pcout << "cg solver called by " << callerFunction << " with "  << solver_control.last_step() << " CG iterations." << std::endl;
  }


  template <int dim>
  void LevelSetEquation<dim>::solve_u()
  {
    LA::MPI::Vector    completely_distributed_solution(locally_owned_dofs,
                                                       mpi_communicator);
    SolverControl            solver_control( dof_handler.n_dofs(), 1e-8 * systemRHS.l2_norm() );
#ifdef USE_PETSC_LA
    LA::SolverGMRES solver(solver_control, mpi_communicator);
#else
    LA::SolverGMRES solver(solver_control);
#endif

    LA::MPI::PreconditionAMG preconditioner;
    LA::MPI::PreconditionAMG::AdditionalData data;
    preconditioner.initialize(systemMatrix, data);

    solver.solve( systemMatrix, 
                  completely_distributed_solution, 
                  systemRHS, 
                  preconditioner );

    pcout << "   u-equation: " << solver_control.last_step() << " GMRES iterations." << std::endl;
    constraints.distribute(completely_distributed_solution);

    solution_u = completely_distributed_solution;
  }
  
  template <int dim>
  void LevelSetEquation<dim>::re_solve_u()
  {
    re_delta_solution_u = 0.0;
    LA::MPI::Vector    completely_distributed_solution( locally_owned_dofs,
                                                        mpi_communicator );

    SolverControl            solver_control( dof_handler.n_dofs(), 1e-8 * systemRHS.l2_norm() );
#ifdef USE_PETSC_LA
    LA::SolverGMRES solver(solver_control, mpi_communicator);
#else
    LA::SolverGMRES solver(solver_control);
#endif

    LA::MPI::PreconditionAMG preconditioner;
    LA::MPI::PreconditionAMG::AdditionalData data;
    preconditioner.initialize(systemMatrix, data);
    
    LA::MPI::Vector    re_solution_u_temp( locally_owned_dofs,
                                           mpi_communicator );
    re_solution_u_temp = re_solution_u;

    completely_distributed_solution = 0.0;

    solver.solve( systemMatrix_re, 
                  completely_distributed_solution, 
                  systemRHS, 
                  preconditioner );
    
    re_delta_solution_u = completely_distributed_solution;
    re_solution_u_temp += completely_distributed_solution;
    re_solution_u       = re_solution_u_temp;
  }
  
  //template <int dim>
  //void LevelSetEquation<dim>::computeAdvection(TensorFunction<1, dim> &AdvectionField_)
  //{
    //std::map<types::global_dof_index, Point<dim> > supportPoints;
    //DoFTools::map_dofs_to_support_points<dim,dim>(MappingQGeneric<dim>(fe.degree),dof_handler,supportPoints);

    //for(auto& globalDofAndNodeCoord : supportPoints)
    //{
        //auto a = AdvectionField_.value(globalDofAndNodeCoord.second);
        //advection_x[globalDofAndNodeCoord.first] = a[0];
        //advection_y[globalDofAndNodeCoord.first] = a[1];
    //} 
  //}

  //// @ to be rearranged
  //template <int dim>
  //void LevelSetEquation<dim>::computeVolume( )
  //{
    //FEValues<dim> fe_values( fe,
                             //qGauss,
                             //update_values | update_gradients | update_JxW_values | update_quadrature_points );

    //const unsigned int dofs_per_cell =   fe.dofs_per_cell;
    //std::vector<types::global_dof_index> globalDofIndices( dofs_per_cell );
    
    //double volumeOfPhase1 = 0;
    //double volumeOfPhase2 = 0;
    //const unsigned int n_q_points    = qGauss.size();
    //std::vector<double> phiAtQ(  n_q_points );
    
    //for (const auto &cell : dof_handler.active_cell_iterators())
    //{
        //fe_values.reinit(               cell );
        //cell->get_dof_indices(          globalDofIndices );
        //fe_values.get_function_values(  solution_u, phiAtQ ); // compute values of old solution

        //for (const unsigned int q_index : fe_values.quadrature_point_indices())
        //{
            //volumeOfPhase1 += ( phiAtQ[q_index] * 0.5 + 0.5 ) * fe_values.JxW(q_index); 
            //volumeOfPhase2 += ( 1. - ( phiAtQ[q_index] * 0.5 + 0.5 ) ) * fe_values.JxW(q_index);
        //}
    //}
    //volumeOfPhase1PerTimeStep.push_back(volumeOfPhase1);
    //volumeOfPhase2PerTimeStep.push_back(volumeOfPhase2);
  //}

  template <int dim>
  void LevelSetEquation<dim>::output_results(const double timeStep)
  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution_u,                      "phi");
    //data_out.add_data_vector(advection_x ,                    "a_x");
    //data_out.add_data_vector(advection_y,                     "a_y");
    //data_out.add_data_vector(solution_normal_damped.block(0), "normal_x");
    //data_out.add_data_vector(solution_normal_damped.block(1), "normal_y");
    //data_out.add_data_vector(solution_curvature_damped,       "curvature");
     Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");
 
    data_out.build_patches();
    data_out.write_vtu_with_pvtu_record(
        "./", "solution", timeStep, mpi_communicator, 2, 8);
    //const std::string filename = "solution-" + Utilities::int_to_string(timeStep, 3) + ".vtu";
    //DataOutBase::VtkFlags vtk_flags;
    //vtk_flags.compression_level = DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
    //data_out.set_flags(vtk_flags);
    //std::ofstream output(filename);
    //data_out.write_vtu(output);
    
  }
  
  template <int dim>
  void LevelSetEquation<dim>::run( const Function<dim>& InitialValues,
                                   TensorFunction<1, dim>& AdvectionField_,
                                   const Function<dim>& DirichletValues) 
  {
    pcout << "Running with "
#ifdef USE_PETSC_LA
          << "PETSc"
#else
          << "Trilinos"
#endif
          << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    pcout << "setup system " << std::endl;
    setup_system(DirichletValues);
    setInitialConditions(InitialValues);
    pcout << "initial conditions set " << std::endl;
    //timeVector.push_back(0.0);

    timestep_number=0;
    //computeAdvection(AdvectionField_);
    //computeDampedNormalLevelSet();
    //computeDampedCurvatureLevelSet();

    output_results( timestep_number );    // print initial state
    //computeVolume( );

    timestep_number++; 
    for (; time <= parameters.end_time; time += parameters.time_step_size, ++timestep_number)
      {
        //timeVector.push_back(time);
        pcout << "Time step " << timestep_number << " at t=" << time << std::endl;

        assembleSystemMatrices(AdvectionField_, DirichletValues); // @todo: insert updateFlag
        
        solve_u();

        if ( parameters.activate_reinitialization )    
        {
            
            computeDampedNormalLevelSet();
            re_solution_u                   = solution_u;
            const double re_time_step       = 1./(dim*dim) * GridTools::minimal_cell_diameter(triangulation); 
            unsigned int re_timestep_number = 0;
            double re_time                  = 0.0;
            
            pcout << "       >>>>>>>>>>>>>>>>>>> REINITIALIZATION START " << std::endl;
            for (; re_timestep_number < parameters.max_n_reinit_steps; re_time += re_time_step, ++re_timestep_number) // 3 to 5 timesteps are enough to reach steady state according to Kronbichler et al.
            {
                computeReinitializationMatrices(re_time_step);
                re_solve_u();
                pcout << "      | Time step " << re_timestep_number << " at tau=" << std::fixed << std::setprecision(5) << re_time << "\t |R|∞ = " << re_delta_solution_u.linfty_norm() << std::endl;
                
                if (re_delta_solution_u.linfty_norm() < 1e-6)
                    break;
            }
            pcout << "       <<<<<<<<<<<<<<<<<<< REINITIALIZATION END " << std::endl;

            solution_u     = re_solution_u;
        }

        //computeAdvection(AdvectionField_);
        //computeDampedNormalLevelSet();
        //computeDampedCurvatureLevelSet();
        output_results(timestep_number);
        
        //if (parameters.compute_volume)
        //{
            //std::cout << " n = " << timestep_number << std::endl;
            //computeVolume();
            //std::cout << "curr volume -- phase 1 " << volumeOfPhase1PerTimeStep[timestep_number] << std::endl;
            //std::cout << "curr volume -- phase 2 " << volumeOfPhase2PerTimeStep[timestep_number] << std::endl;

        //}   
    }

  }
} // end of namespace LevelSet
