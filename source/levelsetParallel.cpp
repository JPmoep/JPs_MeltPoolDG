/* ---------------------------------------------------------------------
 * Author: Magdalena Schreter, TUM, 2020
 */
#include <deal.II/lac/solver_gmres.h>

#include <levelsetParallel.hpp>
#include <reinitialization.hpp>
#include <curvature.hpp>

namespace LevelSetParallel
{
  using namespace dealii; 

  template <int dim, int degree>
  LevelSetEquation<dim,degree>::LevelSetEquation(
                     //parallel::distributed::Triangulation<dim>&       triangulation_,
                     std::shared_ptr<SimulationBase<dim>>             base_in
                     //MPI_Comm&                                        mpi_commun
                     )
    : epsilon ( GridTools::minimal_cell_diameter(base_in->triangulation) / (std::sqrt(dim) * 2.) )
    , mpi_communicator( base_in->mpi_communicator)
    , parameters(       base_in->parameters )
    , fe(               parameters.levelset_degree )
    , triangulation(    base_in->triangulation )
    , dof_handler(      base_in->triangulation )
    , qGauss(           QGauss<dim>(parameters.levelset_degree+1) )
    , time_step(        parameters.time_step_size )
    , time(             parameters.start_time )
    , timestep_number(  1 )
    , pcout(            std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator) == 0) )
    , computing_timer(  mpi_communicator,
                        pcout,
                        TimerOutput::summary,
                        TimerOutput::wall_times)
    , timer(            mpi_communicator)
    , volume_fraction(2,0)
    , field_conditions(    base_in->get_field_conditions()  )
    , boundary_conditions( base_in->get_boundary_conditions()  )
    , reini( mpi_communicator )
    , curvature( mpi_communicator )
  {}
  
  template <int dim, int degree>
  void LevelSetEquation<dim,degree>::print_me() 
  {  
    //if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      //parameters.print_parameters();
    //pcout << "Number of active cells: "       << triangulation.n_active_cells() << std::endl;
    //pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()           << std::endl << std::endl;
  }

  template <int dim, int degree>
  void LevelSetEquation<dim,degree>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");
    dof_handler.distribute_dofs( fe );

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    
                                      
    
    systemRHS.reinit(locally_owned_dofs, 
                     locally_relevant_dofs, 
                     mpi_communicator); 
    
    solution_u.reinit(locally_owned_dofs,
                      locally_relevant_dofs,
                      mpi_communicator);
    
    // constraints for level set function
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    for (auto const& bc : boundary_conditions->dirichlet_bc)
    {
      pcout << "apply dirichlet condition for all faces with" << bc.first << std::endl;
      VectorTools::interpolate_boundary_values( dof_handler,
                                                bc.first,
                                                *bc.second,
                                                constraints );
    }
    constraints.close();   

    //DynamicSparsityPattern dsp( locally_relevant_dofs );
    //DoFTools::make_sparsity_pattern( dof_handler, dsp, constraints, false );
    //SparsityTools::distribute_sparsity_pattern(dsp,
                                               //locally_owned_dofs, 
                                               //mpi_communicator,
                                               //locally_relevant_dofs);

    TrilinosWrappers::SparsityPattern dsp(locally_owned_dofs,
                                         locally_owned_dofs,
                                         locally_relevant_dofs,
                                         mpi_communicator);
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    false,
                                    Utilities::MPI::this_mpi_process(
                                      mpi_communicator));
    dsp.compress();

    systemMatrix.reinit( dsp ); 
    //locally_owned_dofs,
                         //locally_owned_dofs,
                         //dsp,
                         //mpi_communicator);

    //systemMatrix.reinit( locally_owned_dofs,
                         //locally_owned_dofs,
                         //dsp,
                         //mpi_communicator);
                         //
    //@ replace by merge

    constraints_no_dirichlet.clear();
    constraints_no_dirichlet.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints_no_dirichlet); 
    constraints_no_dirichlet.close();   
  }
  

  template <int dim, int degree>
  void LevelSetEquation<dim,degree>::initialize_levelset()
  {
    VectorType solutionTemp( locally_owned_dofs, mpi_communicator);

    VectorTools::project( dof_handler, 
                          constraints,
                          qGauss,
                          *field_conditions->initial_field,           
                          solutionTemp );

    solution_u = solutionTemp;
    solution_u.update_ghost_values();
  } 

  template <int dim, int degree>
  void LevelSetEquation<dim,degree>::assemble_levelset_system()
  {
    TimerOutput::Scope t(computing_timer, "assembly");   
    field_conditions->advection_field->set_time(time);
    
    systemMatrix = 0.0;
    systemRHS    = 0.0;

    FEValues<dim> fe_values( fe,
                             qGauss,
                             update_values | update_gradients | update_JxW_values | update_quadrature_points );

    const unsigned int dofs_per_cell =   fe.dofs_per_cell;
    const unsigned int n_q_points    =   qGauss.size();

    
    std::vector<double>         phiAtQ(  n_q_points );
    std::vector<Tensor<1,dim>>  phiGradAtQ( n_q_points, Tensor<1,dim>() );

    FullMatrix<double> cellMatrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cellRHS( dofs_per_cell );
    
    for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
    {
        std::vector<types::global_dof_index> local_dof_indices( dofs_per_cell );
        
        cellMatrix = 0;
        cellRHS    = 0;

        fe_values.reinit(cell);
        fe_values.get_function_values(     solution_u, phiAtQ ); // compute values of old solution
        fe_values.get_function_gradients(  solution_u, phiGradAtQ ); // compute values of old solution

        for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
            auto qCoord = fe_values.get_quadrature_points()[q_index];
            const Tensor<1, dim> a =  field_conditions->advection_field->value( qCoord );


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
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cellMatrix,
                                               cellRHS,
                                               local_dof_indices,
                                               systemMatrix,
                                               systemRHS);
         
      }
      systemMatrix.compress(VectorOperation::add);
      systemRHS.compress(VectorOperation::add);
  }


  template <int dim, int degree>
  void LevelSetEquation<dim,degree>::initialize_reinitialization_model()
  {
    ReinitializationData reinit_data;
    reinit_data.reinit_model        = ReinitModelType::olsson2007;
    reinit_data.d_tau               = GridTools::minimal_cell_diameter(triangulation);
    reinit_data.degree              = parameters.levelset_degree;
    reinit_data.verbosity_level     = utilityFunctions::VerbosityType::major;
    reinit_data.min_cell_size       = GridTools::minimal_cell_diameter(triangulation);
    reinit_data.do_print_l2norm     = parameters.output_norm_levelset;
    
    DynamicSparsityPattern dsp_re( locally_relevant_dofs );
    DoFTools::make_sparsity_pattern( dof_handler, dsp_re, constraints_no_dirichlet, false );
    SparsityTools::distribute_sparsity_pattern(dsp_re,
                                               locally_owned_dofs,
                                               mpi_communicator,
                                               locally_relevant_dofs);
  
    reini.initialize( reinit_data , 
                      dsp_re,
                      dof_handler,
                      constraints_no_dirichlet,
                      locally_owned_dofs,
                      locally_relevant_dofs );
  }

  template <int dim, int degree>
  void LevelSetEquation<dim,degree>::compute_reinitialization_model()
  {
    // update the solution vector to the reinitialized value
    reini.solve( solution_u );
  }

  template <int dim, int degree>
  void LevelSetEquation<dim,degree>::initialize_curvature()
  {
    CurvatureData curvature_data;
    curvature_data.degree              = parameters.levelset_degree;
    curvature_data.verbosity_level     = utilityFunctions::VerbosityType::major;
    
    DynamicSparsityPattern dsp_re( locally_relevant_dofs );
    DoFTools::make_sparsity_pattern( dof_handler, dsp_re, constraints_no_dirichlet, false );
    SparsityTools::distribute_sparsity_pattern(dsp_re,
                                               locally_owned_dofs,
                                               mpi_communicator,
                                               locally_relevant_dofs);
  
    curvature.initialize( curvature_data , 
                          dsp_re,
                          dof_handler,
                          constraints_no_dirichlet,
                          locally_owned_dofs,
                          locally_relevant_dofs );
  }
  
  template <int dim, int degree>
  void LevelSetEquation<dim,degree>::compute_curvature()
  {
    curvature.solve( solution_u );
  }
  
  template <int dim, int degree>
  void LevelSetEquation<dim,degree>::solve_u()
  {
    TimerOutput::Scope t(computing_timer, "solve");
    VectorType    completely_distributed_solution(locally_owned_dofs,
                                                       mpi_communicator);
    SolverControl            solver_control( dof_handler.n_dofs(), 1e-8 * systemRHS.l2_norm() );
    TrilinosWrappers::PreconditionAMG preconditioner;
    TrilinosWrappers::PreconditionAMG::AdditionalData data;
    
    SolverGMRES<VectorType> solver(solver_control); // mpi_communicator);

    preconditioner.initialize(systemMatrix, data);

    solver.solve( systemMatrix, 
                  completely_distributed_solution, 
                  systemRHS, 
                  PreconditionIdentity());
                  //preconditioner );

    pcout << "   u-equation: " << solver_control.last_step() << " GMRES iterations." << std::endl;
    constraints.distribute(completely_distributed_solution);

    solution_u = completely_distributed_solution;
    solution_u.update_ghost_values();
  }

  //// @ to be rearranged
  template <int dim, int degree>
  void LevelSetEquation<dim,degree>::compute_overall_phase_volume( )
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

  template <int dim, int degree>
  void LevelSetEquation<dim,degree>::output_results( const double timeStep )
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
        
        //data_out.add_data_vector(normal_vector_field.block(0), "normal_x");
        //data_out.add_data_vector(normal_vector_field.block(1), "normal_y");
        //data_out.add_data_vector(curvature_field,              "curvature");
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
  
  template <int dim, int degree>
  void LevelSetEquation<dim,degree>::compute_error( const Function<dim>& ExactSolution )
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
  
  template <int dim, int degree>
  void LevelSetEquation<dim,degree>::run( )
  {
    timer.start();
    pcout << "Running "
          << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    timestep_number=0;
    
    setup_system( );

    initialize_levelset();

    if ( parameters.activate_reinitialization )    
        initialize_reinitialization_model();
    
    output_results( timestep_number );    // print initial state

    timestep_number++; 
    const int p = parameters.levelset_degree;
    const double CFL = 0.2/(p*p);
    const double dx = GridTools::minimal_cell_diameter(triangulation) / std::sqrt(dim);

    //time_step = std::min(parameters.time_step_size, CFL * epsilon /solution_u.max());

    for ( time = time_step; time <= parameters.end_time; time += time_step, ++timestep_number )
    {
        pcout << "Time step " << timestep_number << " at current t=" << time << std::endl;
        Point<2> center     = Point<2>(0.0,0.5);  
        pcout << "calculate advection_field: " << std::endl;
        const Tensor<1, dim> a =  field_conditions->advection_field->value( center );
        pcout << "a[0] " << a[0] << "a[1] " << a[1] << std::endl;
        pcout << "end of calculate advection_field: " << std::endl;

        assemble_levelset_system(  ); // @todo: insert updateFlag
        solve_u();
        if (parameters.output_norm_levelset)
            pcout << " (not reinitialized) levelset function ||phi|| = " << solution_u.l2_norm() << std::endl;
        if ( parameters.activate_reinitialization )    
            compute_reinitialization_model();

        if (parameters.output_norm_levelset)
            pcout << " (reinitialized) levelset function ||phi|| = " << solution_u.l2_norm() << std::endl;

        output_results(timestep_number);

        if ( !parameters.output_walltime )
            computing_timer.disable_output();
        else
        {
            computing_timer.print_summary();
            pcout << "+" << std::string(70, '-') << "+" << std::endl;
            pcout << "| real total wall clock time elapsed since start: " << timer.wall_time() << " s" << std::endl;
            pcout << "+" << std::string(70, '-') << "+" << std::endl;
            computing_timer.reset();  
        } 

        if ( ( time+time_step ) > parameters.end_time)
        {
            time_step = parameters.end_time - time;
            if ( time_step == 0.0 )
                break;
        }
    }
  }
  
  // instantiation
  template class LevelSetEquation<2,1>; 
  template class LevelSetEquation<2,2>; 
  //template class LevelSetEquation<3>;
  //
} // end of namespace LevelSetParallel
