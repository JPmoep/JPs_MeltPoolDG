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
#include <experimental/filesystem>
#include "levelsetparameters.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
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
              TensorFunction<1, dim> &AdvectionField_ 
            );

  private:
    void setup_system();
    void setInitialConditions(const Function<dim>& InitialValues);
    void setDirichletBoundaryConditions();
    void assembleSystemMatrices(TensorFunction<1, dim> &AdvectionField_);
    void computeReinitializationMatrices( const double dTau );
    void solve_u();
    void re_solve_u();
    void output_results(const double timeStep);
    void computeVolume();
    void computeAdvection(TensorFunction<1, dim> &AdvectionField_);
    //void computeDampedNormalLevelSet();
    
    Triangulation<dim>&        triangulation;
    LevelSetParameters        parameters;
    FE_Q<dim>                 fe;
    DoFHandler<dim>           dof_handler;
    QGauss<dim>               qGauss; 
    AffineConstraints<double> constraints;

    SparsityPattern    sparsity_pattern;

    SparseMatrix<double> systemMatrix; // global system matrix
    Vector<double>       systemRHS; // global system matrix

    Vector<double> old_solution_u;
    Vector<double>     solution_u;
    Vector<double> re_solution_u;
    Vector<double> re_delta_solution_u;
    Vector<double> advection_x;
    Vector<double> advection_y;
    
    std::vector<double> volumeOfPhase1PerTimeStep;
    std::vector<double> volumeOfPhase2PerTimeStep;
    std::vector<double> timeVector;

    double       time_step;
    double       time;
    unsigned int timestep_number;
    //const double theta;
  };


  template <int dim>
  LevelSetEquation<dim>::LevelSetEquation(
                     const LevelSetParameters& parameters_,
                     Triangulation<dim>&       triangulation_)
    : fe(               parameters.levelSetDegree )
    , triangulation(    triangulation_ )
    , dof_handler(      triangulation_ )
    , parameters(       parameters_)
    , qGauss(           QGauss<dim>(parameters_.levelSetDegree+1) )
    , time_step(        parameters_.timeStep )
    , time(             time_step )
    , timestep_number(  1 )
    //, theta(            parameters_.theta )  // 0 = explicit euler, 0.5 = Crank-Nicolson, 1.0 = implicit euler
  {}
  
  template <int dim>
  void LevelSetEquation<dim>::setInitialConditions(const Function<dim>& InitialValues)
  {
    Point<2> center     = Point<2>(0,0.5);
    VectorTools::project(dof_handler, 
                         constraints,
                         qGauss,
                         InitialValues,           
                         solution_u);
  } 
  
  template <int dim>
  void LevelSetEquation<dim>::setDirichletBoundaryConditions()
  {
    std::cout << "dirichlet values set " << parameters.dirichletBoundaryValue << std::endl;
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values( dof_handler,
                                              utilityFunctions::BCTypes::dirichlet,
                                              Functions::ConstantFunction<dim>( parameters.dirichletBoundaryValue ),
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
                                             parameters.theta * time_step * ( parameters.diffusivity * 
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
                              parameters.diffusivity *
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

    setDirichletBoundaryConditions(); 
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
            const double diffRhs = parameters.epsInterface * normalAtQ[q_index] * psiGradAtQ[q_index];

            for (const unsigned int i : fe_values.dof_indices())
            {
                const double nTimesGradient_i = normalAtQ[q_index] * fe_values.shape_grad(i, q_index);

                for (const unsigned int j : fe_values.dof_indices())
                {
                    const double nTimesGradient_j = normalAtQ[q_index] * fe_values.shape_grad(j, q_index);
                    cell_matrix(i,j) += (
                                          fe_values.shape_value(i,q_index) * fe_values.shape_value(j,q_index)
                                          + 
                                          dTau * parameters.epsInterface * nTimesGradient_i * nTimesGradient_j
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

  //template <int dim>
  //void LevelSetEquation<dim>::computeDampedNormalLevelSet()
  //{
    
    //auto qGauss = QGauss<dim>(fe.degree + 1); 

    //const unsigned int nComp = 2;
    //FESystem<dim> feNormals(fe, 2);
    //FEValues<dim> fe_values( fe,
                             //qGauss,
                             //update_values | update_gradients | update_quadrature_points | update_JxW_values );

    //const unsigned int   dofs_per_cell = feNormals.dofs_per_cell;
    //FullMatrix<double>   normal_cell_matrix( dofs_per_cell, dofs_per_cell );
    //Vector<double>       normal_cell_rhs(    dofs_per_cell );
    
    //std::vector<types::global_dof_index> globalDofIndices( dofs_per_cell );

    //for (const auto &cell : dof_handler.active_cell_iterators())
    //{
        //fe_values.reinit(cell);
        
        //normal_cell_matrix = 0.0;
        //normal_cell_rhs =    0.0;
        
        //for (const unsigned int q_index : fe_values.quadrature_point_indices())
        //{
            //for (unsigned int i=0; i<dofs_per_cell; ++i)
                //for (unsigned int j=0; j<dofs_per_cell; ++j)
                    //if( feNormals.system_to_component_index(i).first == feNormals.system_to_component_index(j).first)
                    //{ 
                        //const unsigned int shape_i = feNormals.system_to_component_index(i).second;
                        //const unsigned int shape_j = feNormals.system_to_component_index(j).second;
                        //normal_cell_matrix(i,j) += fe_values.shape_value(shape_i,q_index) * fe_values.shape_value(shape_j,q_index) *fe_values.JxW( q_index ) ;

                    //}
        //}

        //normal_cell_matrix.print(std::cout);
        //normal_cell_matrix = 0.0;
        //std::cout << " -------- ALTERNATIV ---------------" << std::endl;
         ////ALTERNATIVE
        //for (const unsigned int q_index : fe_values.quadrature_point_indices())
        //{
            //for (unsigned int d=0; d<dim; ++d)
                //for (unsigned int i=d; i<dofs_per_cell; i+=nComp)
                    //for (unsigned int j=d; j<dofs_per_cell; j+=nComp)
                    //{
                        //const unsigned int k_j = (j - d) / nComp ;
                        //const unsigned int k_i = (i - d) / nComp ;
                        //normal_cell_matrix(i,j) += fe_values.shape_value(k_i,q_index) * fe_values.shape_value(k_j,q_index) * fe_values.JxW( q_index ) ;
                        //std::cout << " ( " << i << "," << j << "): N_" << k_i << " * N_" << k_j << std::endl;
                    //}
        //}
        //normal_cell_matrix.print(std::cout);


        //for (const unsigned int i : feNormals_values.dof_indices())
        //{
          //const unsigned int component_i = feNormals.system_to_component_index(i).first;
          
          //std::cout << "comp_i " << component_i << std::endl;

          //////for (const unsigned int j : fe_values.dof_indices())
            //////{
              //////const unsigned int component_j = fe.system_to_component_index(j).first;

                //////{
                  //////cell_matrix(i, j) +=
        //}
    //}
  //}


  template <int dim>
  void LevelSetEquation<dim>::setup_system()
  {

    //GridGenerator::hyper_cube( triangulation, InputParameters::leftDomain, InputParameters::rightDomain );

    //// mark left edge as inflow boundaryCondition
    ////for (auto &face : triangulation.active_face_iterators())
      ////if ( (face->at_boundary() ) ) // && (face->center()[0] == InputParameters::leftDomain) )
      ////{
          ////face->set_boundary_id ( utilityFunctions::BCTypes::inflow );
      ////}

    //triangulation.refine_global( InputParameters::nMeshRefinements );
    dof_handler.distribute_dofs( fe );

    std::cout << "Number of active cells: "       << triangulation.n_active_cells() << std::endl;
    std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()           << std::endl << std::endl;

    DynamicSparsityPattern dsp(      dof_handler.n_dofs(), dof_handler.n_dofs() );
    DoFTools::make_sparsity_pattern( dof_handler, dsp );
    sparsity_pattern.copy_from(      dsp );
    
    systemMatrix.reinit(             sparsity_pattern );
                                      
    systemRHS.reinit(                dof_handler.n_dofs() );
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
    const unsigned int dofs_per_cell =   fe.dofs_per_cell;
    
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
    data_out.add_data_vector(solution_u, "U");
    data_out.add_data_vector(advection_x , "a_x");
    data_out.add_data_vector(advection_y,  "a_y");
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
                                   TensorFunction<1, dim>& AdvectionField_) 
  {
    std::cout << "setup system " << std::endl;
    setup_system();
    std::cout << "initial conditions set " << std::endl;
    setInitialConditions(InitialValues);
    timeVector.push_back(0.0);

    timestep_number=0;
    computeAdvection(AdvectionField_);
    output_results( timestep_number );    // print initial state
    computeVolume( );

    timestep_number++; 
    for (; time <= parameters.maxTime; time += parameters.timeStep, ++timestep_number)
      {
        timeVector.push_back(time);
        std::cout << "Time step " << timestep_number << " at t=" << time << std::endl;

        assembleSystemMatrices(AdvectionField_); // @todo: insert updateFlag
        
        solve_u();
        
        if ( parameters.activateReinitialization )    
        {
            re_solution_u     = solution_u;
            const double re_time_step       = 1./(dim*dim) * parameters.characteristicMeshSize; 
            unsigned int re_timestep_number = 0;
            double re_time                  = 0.0;
            
            std::cout << "       >>>>>>>>>>>>>>>>>>> REINITIALIZATION START " << std::endl;
            for (; re_timestep_number < 10; re_time += re_time_step, ++re_timestep_number) // 3 to 5 timesteps are enough to reach steady state according to Kronbichler et al.
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
        output_results(timestep_number);
        
        if (parameters.computeVolume)
        {
            std::cout << " n = " << timestep_number << std::endl;
            computeVolume();
                std::cout << "curr volume -- phase 1 " << volumeOfPhase1PerTimeStep[timestep_number] << std::endl;
                std::cout << "curr volume -- phase 2 " << volumeOfPhase2PerTimeStep[timestep_number] << std::endl;

        }   
    }
  }

} // end of namespace LevelSet

// deprecated
        // re-initialization
      //FEEvaluation<dim> feEval(fe, 
                               //QGauss<dim - 1>(fe.degree + 1),
                              //update_values | update_gradients | update_JxW_values | update_normal_vectors);
        //FEEvaluation<dim,fe_degree> phi(matrix_free);
        //for (const auto &cell : dof_handler.active_cell_iterators())
          //{
            //phi.reinit(cell_index);
            //phi.read_dof_values(solution_u);
            //phi.evaluate(true, false);   // interpolate values, but not gradients
            //for (unsigned int q_index=0; q_index<phi.n_q_points; ++q_index)
              //{
                //VectorizedArray<double> val = phi.get_value(q_index);
                //// do something with val
              //}
          //}

