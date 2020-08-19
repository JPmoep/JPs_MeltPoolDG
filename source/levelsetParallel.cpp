/* ---------------------------------------------------------------------
 * Author: Magdalena Schreter, TUM, 2020
 */
#include <levelsetParallel.hpp>
#include <deal.II/lac/generic_linear_algebra.h>

namespace LevelSetParallel
{
  using namespace dealii; 

  template <int dim>
  LevelSetEquation<dim>::LevelSetEquation(
                     const LevelSetParameters& parameters_,
                     parallel::distributed::Triangulation<dim>&       triangulation_,
                     TensorFunction<1, dim>& AdvectionField_,
                     MPI_Comm& mpi_commun)
    : epsilon ( GridTools::minimal_cell_diameter(triangulation_) / (std::sqrt(dim) * 2.) )
    , mpi_communicator( mpi_commun)
    , parameters(       parameters_)
    , fe(               parameters_.levelset_degree )
    , triangulation(    triangulation_ )
    , dof_handler(      triangulation_ )
    , qGauss(           QGauss<dim>(parameters_.levelset_degree+1) )
    , time_step(        parameters_.time_step_size )
    , time(             parameters_.start_time )
    , timestep_number(  1 )
    , pcout(            std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(  mpi_communicator,
                        pcout,
                        TimerOutput::summary,
                        TimerOutput::wall_times)
    , timer(            mpi_communicator)
    , AdvectionField(   AdvectionField_ )
    , volume_fraction(2,0)
  {}
  
  
  template <int dim>
  void LevelSetEquation<dim>::setup_system(const Function<dim>& DirichletValues )
  {
    TimerOutput::Scope t(computing_timer, "setup");
    dof_handler.distribute_dofs( fe );

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    
    pcout << "Number of active cells: "       << triangulation.n_active_cells() << std::endl;
    pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()           << std::endl << std::endl;
                                      
    systemRHS.reinit(locally_owned_dofs, mpi_communicator); 
    
    solution_u.reinit(locally_owned_dofs,
                      locally_relevant_dofs,
                      mpi_communicator);
    
    re_solution_u.reinit( locally_owned_dofs,
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
    
    system_curvature_RHS.reinit( locally_owned_dofs, mpi_communicator ); 
    curvature_field.reinit( locally_owned_dofs, 
                            locally_relevant_dofs,
                            mpi_communicator ); 

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
    // -------------------------------------------------------
    // constraints for reinitialization function
    // -------------------------------------------------------
    constraints_re.clear();
    constraints_re.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints_re); 
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
    solution_u.update_ghost_values();
  } 

  template <int dim>
  void LevelSetEquation<dim>::assemble_levelset_system(const Function<dim>& DirichletValues)
  {
    TimerOutput::Scope t(computing_timer, "assembly");   
    AdvectionField.set_time(time);
    systemMatrix = 0.0;
    systemRHS    = 0.0;

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
    
    
    for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
    {
        cellMatrix = 0;
        cellRHS    = 0;

        fe_values.reinit(cell);
        fe_values.get_function_values(     solution_u, phiAtQ ); // compute values of old solution
        fe_values.get_function_gradients(  solution_u, phiGradAtQ ); // compute values of old solution

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
            auto qCoord = fe_values.get_quadrature_points()[q_index];
            Tensor<1, dim> a = AdvectionField.value( qCoord );

            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                auto velocityGradShape = a * fe_values.shape_grad( i, q_index);  // grad_phi_j(x_q)

                for (unsigned int j=0; j<dofs_per_cell; ++j)
                {
                    auto velocityGradShape_j = a * fe_values.shape_grad( j, q_index);  // grad_phi_j(x_q)
                    //cellMatrix( i, j ) += (  fe_values.shape_value( i, q_index) * 
                                             //fe_values.shape_value( j, q_index) +
                                             //parameters.theta * time_step * ( parameters.artificial_diffusivity * 
                                                                   //fe_values.shape_grad( i, q_index) * 
                                                                   //fe_values.shape_grad( j, q_index) -
                                                                   //velocityGradShape * 
                                                                   //fe_values.shape_value( j, q_index) )
                                          //) * fe_values.JxW(q_index);                                    
                    cellMatrix( i, j ) += (  fe_values.shape_value( i, q_index) * 
                                             fe_values.shape_value( j, q_index) +
                                             parameters.theta * time_step * ( parameters.artificial_diffusivity * 
                                                                   fe_values.shape_grad( i, q_index) * 
                                                                   fe_values.shape_grad( j, q_index) +
                                                                   fe_values.shape_value( i, q_index) ) *
                                                                   velocityGradShape_j 
                                          ) * fe_values.JxW(q_index);                                    
                    
                }
                cellRHS( i ) +=
                   (  fe_values.shape_value( i, q_index) * phiAtQ[q_index]
                       - 
                      ( 1. - parameters.theta ) * time_step * (
                              parameters.artificial_diffusivity *
                              fe_values.shape_grad( i, q_index) *
                              phiGradAtQ[q_index]
                              +
                              a * phiGradAtQ[q_index]    
                              * fe_values.shape_value(  i, q_index)
                         )
                   ) * fe_values.JxW(q_index) ;       // dx

                //cellRHS( i ) +=
                   //(  fe_values.shape_value( i, q_index) * phiAtQ[q_index]
                       //- 
                      //( 1. - parameters.theta ) * time_step * (
                              //parameters.artificial_diffusivity *
                              //fe_values.shape_grad( i, q_index) *
                              //phiGradAtQ[q_index]
                              //-
                              //a * fe_values.shape_grad(  i, q_index)
                              //* phiAtQ[q_index]    
                         //)
                   //) * fe_values.JxW(q_index) ;       // dx
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
  void LevelSetEquation<dim>::assemble_reinitialization_system( )
  {
    pcout << "       >>>>>>>>>>>>>>>>>>> REINITIALIZATION START " << std::endl;
    TimerOutput::Scope t(computing_timer, "reinitialize");

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
    
    std::vector<types::global_dof_index> local_dof_indices( dofs_per_cell );
    
    bool normalsQuick = false;

    const double re_time_step       = GridTools::minimal_cell_diameter(triangulation) / std::sqrt(dim); // * GridTools::minimal_cell_diameter(triangulation) ;
    const double dTau = re_time_step; 
    unsigned int re_timestep_number = 0;
    double re_time                  = dTau;

   for (; re_timestep_number < parameters.max_n_reinit_steps; re_time += re_time_step, ++re_timestep_number) // 3 to 5 timesteps are enough to reach steady state according to Kronbichler et al.
   {
        systemRHS       = 0;
        systemMatrix_re = 0.0;
        if (re_timestep_number == 0 )
        {
            computeNormalLevelSet();
        }
        for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
        {
           cell_matrix = 0.0;
           cell_rhs = 0.0;
           const double epsilon_cell = cell->diameter() / ( std::sqrt(dim) * 2 );
           fe_values.reinit(cell);
           
           fe_values.get_function_values(     solution_u, psiAtQ ); // compute values of old solution
           fe_values.get_function_gradients(  solution_u, psiGradAtQ ); // compute values of old solution

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
                       normalAtQ[q_index][d] = temp[q_index];
               }
               for (auto& n : normalAtQ)
                   n /= n.norm(); //@todo: add exception
           }
           // @todo: only compute normals once during timestepping
           for (const unsigned int q_index : fe_values.quadrature_point_indices())
            {
               const double diffRhs = epsilon_cell * normalAtQ[q_index] * psiGradAtQ[q_index];

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
                                                 dTau * epsilon_cell * nTimesGradient_i * nTimesGradient_j
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
          cell->get_dof_indices(local_dof_indices);
          constraints_re.distribute_local_to_global(cell_matrix,
                                                    cell_rhs,
                                                    local_dof_indices,
                                                    systemMatrix_re,
                                                    systemRHS);
           
        }
        systemMatrix_re.compress( VectorOperation::add );
        systemRHS.compress(    VectorOperation::add );
        
        SolverControl solver_control( dof_handler.n_dofs() , 1e-3 * systemRHS.l2_norm() );
        
        LA::SolverCG solver(solver_control, mpi_communicator);

        LA::MPI::PreconditionAMG preconditioner;
        LA::MPI::PreconditionAMG::AdditionalData data;
        preconditioner.initialize(systemMatrix_re, data);

        LA::MPI::Vector    re_solution_u_temp( locally_owned_dofs,
                                           mpi_communicator );
        re_solution_u_temp = solution_u;
        
        re_delta_solution_u = 0;

        solver.solve( systemMatrix_re, 
                      re_delta_solution_u, 
                      systemRHS, 
                      preconditioner );
        constraints_re.distribute(re_delta_solution_u);

        re_solution_u_temp += re_delta_solution_u;
        
        solution_u = re_solution_u_temp;
        solution_u.update_ghost_values();
        //output_results( re_timestep_number+1);    // print initial state
        
        pcout << "      | Time step " << re_timestep_number << " at tau=" << std::fixed << std::setprecision(5); 
        pcout << re_time << "\t |R|∞ = " << re_delta_solution_u.linfty_norm() << "\t |R|²/dT = " << re_delta_solution_u.l2_norm()/dTau << std::endl;
        pcout << "      | cg solver called by " << "reinitialization "<< " with "  << solver_control.last_step() << " CG iterations." << std::endl;

        if (re_delta_solution_u.l2_norm() / dTau < 1e-6)
           break;
    }
    pcout << "       >>>>>>>>>>>>>>>>>>> REINITIALIZATION END " << std::endl;
  }
  
  template <int dim>
  void LevelSetEquation<dim>::computeNormalLevelSet()
  {
    TimerOutput::Scope t(computing_timer, "compute damped normals");  
    systemMatrix_re = 0.0;
    for (unsigned int d=0; d<dim; d++)
        system_normal_RHS.block(d) = 0.0;
    
    FEValues<dim> fe_values( fe,
                             qGauss,
                             update_values | update_gradients | update_quadrature_points | update_JxW_values );
    const unsigned int n_q_points    = qGauss.size();

    const unsigned int          dofs_per_cell = fe.dofs_per_cell;
    FullMatrix<double>          normal_cell_matrix( dofs_per_cell, dofs_per_cell );
    std::vector<Vector<double>> normal_cell_rhs(    dim, Vector<double>(dofs_per_cell) );
    
    std::vector<types::global_dof_index> local_dof_indices( dofs_per_cell );
    std::vector<Tensor<1,dim>>           normal_at_q(  n_q_points, Tensor<1,dim>() );

    double damping = GridTools::minimal_cell_diameter(triangulation) * 0.5; //@todo: modifiy damping parameter
    
    for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
    {
        fe_values.reinit(cell);
        cell->get_dof_indices( local_dof_indices );
        
        normal_cell_matrix = 0.0;
        for(auto& normal_cell : normal_cell_rhs)
            normal_cell =    0.0;

        fe_values.get_function_gradients( solution_u, normal_at_q ); // compute normals from level set solution at tau=0
        //for (auto& n : normal_at_q)
         //{
             //if (n.norm()<1e-10)
                //std::cout << "@@@@@@@@@@@q NORM IS ZERO " << n.norm() <<std::endl;
             //n /= n.norm();
         //}
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
    {
        solve_cg(system_normal_RHS.block( d ), systemMatrix_re, normal_vector_field.block( d ), "damped normals");
        constraints_re.distribute(normal_vector_field.block( d ));
   }
  }

  template <int dim>
  void LevelSetEquation<dim>::computeCurvatureLevelSet()
  {
    TimerOutput::Scope timer (computing_timer, "Curvature computation.");

    computeNormalLevelSet();
    
    systemMatrix_re = 0.0; 
    system_curvature_RHS = 0.0;

    FEValues<dim> fe_values( fe,
                             qGauss,
                             update_values | update_gradients | update_quadrature_points | update_JxW_values );

    const unsigned int n_q_points    = qGauss.size();
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    
    std::vector<types::global_dof_index> local_dof_indices(     dofs_per_cell );
    FullMatrix<double>                   curvature_cell_matrix( dofs_per_cell, dofs_per_cell );
    Vector<double>                       curvature_cell_rhs(    dofs_per_cell );

    const double curvature_damping = 0.0; //@todo: modifiy damping parameter

    for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
    {
        fe_values.reinit( cell );
        
        curvature_cell_matrix = 0.0;
        curvature_cell_rhs    = 0.0;
 
        std::vector<Tensor<1,dim>>           normal_at_q( n_q_points, Tensor<1,dim>() );
        for (unsigned int d=0; d<dim; ++d)
        {
            std::vector<double> temp(n_q_points);
            
            fe_values.get_function_values( normal_vector_field.block(d), temp ); 

            for (const unsigned int q_index : fe_values.quadrature_point_indices())
                normal_at_q[q_index][d] = temp[q_index];
        }
        
        // ALTERNATIVE
        //fe_values.get_function_gradients( solution_u, normal_at_q); // compute normals from level set solution at tau=0
        //for (auto& n : normal_at_q)
            //n /= n.norm(); //@todo: add exception

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
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
                                                       curvature_damping * grad_phi_i * grad_phi_j  
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
        
        cell->get_dof_indices(local_dof_indices);
        constraints_re.distribute_local_to_global(curvature_cell_matrix,
                                                  curvature_cell_rhs,
                                                  local_dof_indices,
                                                  systemMatrix_re,
                                                  system_curvature_RHS);
    } // end loop over cells
    systemMatrix_re.compress(      VectorOperation::add );
    system_curvature_RHS.compress( VectorOperation::add );

    solve_cg(system_curvature_RHS, systemMatrix_re, curvature_field, "damped curvature");
  }

  template <int dim>
  void LevelSetEquation<dim>::solve_cg(const LA::MPI::Vector&       RHS,
                                       const LA::MPI::SparseMatrix& matrix,
                                       LA::MPI::Vector&             solution,
                                       const std::string&           callerFunction)
  {
    LA::MPI::Vector    completely_distributed_solution(locally_owned_dofs,
                                                       mpi_communicator);
    SolverControl            solver_control( dof_handler.n_dofs() * 2, 1e-6 * RHS.l2_norm() );
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
    TimerOutput::Scope t(computing_timer, "solve");
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
    //solution_u.update_ghost_values();
  }
  
  //@ to be added!!
  //template <int dim>
  //void LevelSetEquation<dim>::computeAdvection( TensorFunction<1, dim> &AdvectionField_)
  //{
    //AdvectionField_.set_time(time);
    
    //std::vector<types::global_dof_index> local_dof_indices( fe.dofs_per_cell );

    //for (const auto &cell : dof_handler.active_cell_iterators())
    ////if(cell->is_locally_owned())
    ////{
        //cell->get_dof_indices(local_dof_indices);
        //for (auto vertex_index : GeometryInfo<dim>::vertex_indices())
        //{
            ////if (locally_owned_dofs.is_element(local_dof_indices[vertex_index]))
            ////{
                //auto a = AdvectionField_.value(cell->vertex(vertex_index)); 
                //for (unsigned int d=0; d<dim; d++)
                    //advection_field.block(d)[local_dof_indices[vertex_index]] = a[d];
            ////} 
            ////else
                ////std::cout << "proc: " << Utilities::MPI::this_mpi_process(mpi_communicator) << "dof" << local_dof_indices[vertex_index] << std::endl;
        //}
    ////}
    //std::cout << " -----------" << std::endl;
  //}

  //// @ to be rearranged
  template <int dim>
  void LevelSetEquation<dim>::compute_overall_phase_volume( )
  {
    Vector<double> phase_volume_per_cell(triangulation.n_active_cells());

    FEValues<dim> fe_values( fe,
                             qGauss,
                             update_values | update_JxW_values | update_quadrature_points );

    const unsigned int dofs_per_cell =   fe.dofs_per_cell;
    
    const unsigned int n_q_points    = qGauss.size();
    std::vector<double> phi_at_q(  n_q_points );

    //const double& max_value = solution_u.max();
    //const double& min_value = solution_u.min();
    const double& max_value = 1.0;
    const double& min_value = -1.0;
    
    std::fill(volume_fraction.begin(), volume_fraction.end(), 0);
    double phaseValue1 = 0;
    double phaseValue2 = 0;
    const double threshhold = 0.5;

    for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
    {
        fe_values.reinit(               cell );
        fe_values.get_function_values(  solution_u, phi_at_q ); // compute values of old solution

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
            const double phi_normalized = utilityFunctions::normalizeFunction ( phi_at_q[q_index], min_value, max_value );
            if (phi_normalized>=threshhold)
                phaseValue1 += fe_values.JxW(q_index);
            else 
                phaseValue2 += fe_values.JxW(q_index);
            //phaseValue1 += ( phi_at_q[q_index] * 0.5 + 0.5 ) * fe_values.JxW(q_index); 
            //phaseValue2 += ( 1. - ( phi_at_q[q_index] * 0.5 + 0.5 ) ) * fe_values.JxW(q_index);

        }
    }
    volume_fraction[0] = Utilities::MPI::sum(phaseValue1, mpi_communicator);
    volume_fraction[1] = Utilities::MPI::sum(phaseValue2, mpi_communicator);

    //@ todo: write template class for formatted output table
    std::cout << "vol 1: " << volume_fraction[0] << "vol 2: " << volume_fraction[1] << std::endl;
    size_t headerWidths[2] = {
        std::string("time").size(),
        std::string("volume phase 1").size(),
    };

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        if ( time==parameters.start_time )
        {
            std::cout << "output file opened" << std::endl;
            std::fstream fs;
            fs.open (parameters.filename_volume_output, std::fstream::out);
            fs.precision(10);
            fs << "time | volume phase 1 | volume phase 2 " << std::endl; 
            fs << std::left << std::setw(headerWidths[0]) << time;
            fs << "   " << std::left << std::setw(headerWidths[1]) << volume_fraction[0]; 
            fs << "   " << std::left << std::setw(headerWidths[1]) << volume_fraction[1] << std::endl; 
            fs.close();
        }
        else
        {
            std::fstream fs;
            fs.open (parameters.filename_volume_output,std::fstream::in | std::fstream::out | std::fstream::app);
            fs.precision(10);
            fs << std::left << std::setw(headerWidths[0]) << time;
            fs << "   " << std::left << std::setw(headerWidths[1]) << volume_fraction[0]; 
            fs << "   " << std::left << std::setw(headerWidths[1]) << volume_fraction[1] << std::endl; 
        }
    }
  }

  template <int dim>
  void LevelSetEquation<dim>::output_results( const double timeStep )
  {
    if (parameters.compute_paraview_output)
    {
        TimerOutput::Scope t(computing_timer, "output_results");   

        std::vector<std::string> solution_names(dim, "velocity");

        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);

        utilityFunctions::GradientPostprocessor<dim>    gradient_postprocessor;

        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution_u,                      "phi");

        data_out.add_data_vector(solution_u, gradient_postprocessor);
        
        data_out.add_data_vector(normal_vector_field.block(0), "normal_x");
        data_out.add_data_vector(normal_vector_field.block(1), "normal_y");
        data_out.add_data_vector(curvature_field,              "curvature");
         Vector<float> subdomain(triangulation.n_active_cells());
        for (unsigned int i = 0; i < subdomain.size(); ++i)
          subdomain(i) = triangulation.locally_owned_subdomain();
        data_out.add_data_vector(subdomain, "subdomain");
     
        data_out.build_patches();

        data_out.write_vtu_with_pvtu_record(
            "./", parameters.filename_paraview_output, timeStep, mpi_communicator, 2, 8);
    }
    if (parameters.compute_volume_output)
        compute_overall_phase_volume();
  }
  
  template <int dim>
  void LevelSetEquation<dim>::compute_error( const Function<dim>& ExactSolution )
  {
        Vector<double> norm_per_cell(triangulation.n_active_cells());

        VectorTools::integrate_difference(dof_handler,
                                          solution_u,
                                          ExactSolution,
                                          norm_per_cell,
                                          qGauss,
                                          VectorTools::L2_norm);
        
        pcout     << "L2 error =    "
                  << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                  << compute_global_error(triangulation, 
                                          norm_per_cell,
                                          VectorTools::L2_norm) << std::endl;

        Vector<float> difference_per_cell(triangulation.n_active_cells());

        VectorTools::integrate_difference(dof_handler,
                                          solution_u,
                                          ExactSolution,
                                          difference_per_cell,
                                          qGauss,
                                          VectorTools::L1_norm);

        double h1_error = VectorTools::compute_global_error(triangulation,
                                                            difference_per_cell,
                                                            VectorTools::L1_norm);
        pcout << "L1 error = " << h1_error << std::endl;
        
  }
  
  template <int dim>
  void LevelSetEquation<dim>::run( const Function<dim>& InitialValues,
                                   const Function<dim>& DirichletValues) 
  {
    timer.start();
    pcout << "Running with "
#ifdef USE_PETSC_LA
          << "PETSc"
#else
          << "Trilinos"
#endif
          << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    pcout << "setup system " << std::endl;
    timestep_number=0;
    setup_system( DirichletValues );
    setInitialConditions(InitialValues);
    if ( parameters.activate_reinitialization )    
        assemble_reinitialization_system();
    
    output_results( timestep_number );    // print initial state

    timestep_number++; 
    const int p = parameters.levelset_degree;
    const double CFL = 0.2/(p*p);
    const double dx = GridTools::minimal_cell_diameter(triangulation) / std::sqrt(dim);

    time_step = std::min(parameters.time_step_size, CFL*epsilon/solution_u.max());
    
    for ( time = time_step; time <= parameters.end_time; time += time_step, ++timestep_number )
    {
        pcout << "Time step " << timestep_number << " at current t=" << time << std::endl;

        assemble_levelset_system(  DirichletValues ); // @todo: insert updateFlag
        solve_u();
        if ( parameters.activate_reinitialization )    
            assemble_reinitialization_system();

        output_results(timestep_number);

        computing_timer.print_summary();
        pcout << "+" << std::string(70, '-') << "+" << std::endl;
        pcout << "| real total wall clock time elapsed since start: " << timer.wall_time() << " s" << std::endl;
        pcout << "+" << std::string(70, '-') << "+" << std::endl;
        computing_timer.reset();    
        if ( ( time+time_step ) > parameters.end_time)
        {
            time_step = parameters.end_time - time;
            if ( time_step == 0.0 )
                break;
        }
    }
  }
  
  // instantiation
  template class LevelSetEquation<2>; 
  template class LevelSetEquation<3>;
} // end of namespace LevelSet
