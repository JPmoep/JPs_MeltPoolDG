/* ---------------------------------------------------------------------
 * Author: Magdalena Schreter, TUM, 2020
 */
#include <deal.II/lac/solver_gmres.h>

#include <levelsetParallel.hpp>
#include <linearsolve.hpp>
#include <reinitialization.hpp>
#include <curvature.hpp>

namespace LevelSetParallel
{
  using namespace dealii; 

  template <int dim, int degree>
  LevelSetEquation<dim,degree>::LevelSetEquation( std::shared_ptr<SimulationBase<dim>> base_in )
    : epsilon (            GridTools::minimal_cell_diameter(base_in->triangulation) / ( std::sqrt(dim) * 2. ) ) // @todo: is this variable really necessary?
    , mpi_communicator(    base_in->mpi_communicator)
    , parameters(          base_in->parameters )
    , fe(                  parameters.levelset_degree )
    , triangulation(       base_in->triangulation )
    , dof_handler(         base_in->triangulation )
    , pcout(               std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator) == 0) )
    , computing_timer(     mpi_communicator,
                           pcout,
                           TimerOutput::summary,
                           TimerOutput::wall_times)
    , timer(               mpi_communicator)
    , field_conditions(    base_in->get_field_conditions()  )
    , boundary_conditions( base_in->get_boundary_conditions()  )
    , reini(               mpi_communicator )
    , curvature(           mpi_communicator )
  {}
  
  template <int dim, int degree>
  void LevelSetEquation<dim,degree>::print_me() 
  {  
    pcout << "Number of active cells: "       << triangulation.n_active_cells() << std::endl;
    pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()           << std::endl << std::endl;
  }

  template <int dim, int degree>
  void LevelSetEquation<dim,degree>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");
    dof_handler.distribute_dofs( fe );

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    
                                      
    
    system_rhs.reinit(locally_owned_dofs, 
                     locally_relevant_dofs, 
                     mpi_communicator); 
    
    solution_levelset.reinit(locally_owned_dofs,
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


    // distributed sparsity pattern --> not required for matrix-free solution algorithm 
    TrilinosWrappers::SparsityPattern dsp( locally_owned_dofs,
                                           locally_owned_dofs,
                                           locally_relevant_dofs,
                                           mpi_communicator);
    DoFTools::make_sparsity_pattern( dof_handler,
                                     dsp,
                                     constraints,
                                     false,
                                     Utilities::MPI::this_mpi_process(mpi_communicator)
                                   );
    dsp.compress();

    system_matrix.reinit( dsp ); 

    // constraints for subproblem classes --> no dirichlet conditions
    constraints_no_dirichlet.clear();
    constraints_no_dirichlet.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints_no_dirichlet); 
    constraints_no_dirichlet.close();   
  }
  

  template <int dim, int degree>
  void LevelSetEquation<dim,degree>::initialize_levelset()
  {
    setup_system();
    // @ is there a better solution to avoid local copy??
    VectorType solutionTemp( locally_owned_dofs, mpi_communicator);

    VectorTools::project( dof_handler, 
                          constraints,
                          QGauss<dim>(parameters.levelset_degree+1),
                          *field_conditions->initial_field,           
                          solutionTemp );

    solution_levelset = solutionTemp;
    solution_levelset.update_ghost_values();
  } 

  template <int dim, int degree>
  void LevelSetEquation<dim,degree>::compute_levelset_model()
  {
    TimerOutput::Scope t(computing_timer, "assembly");   
    field_conditions->advection_field->set_time( time_iterator.get_current_time() );
    
    system_matrix = 0.0;
    system_rhs    = 0.0;
    
    const auto qGauss = QGauss<dim>(parameters.levelset_degree+1);
    
    FEValues<dim> fe_values( fe,
                             qGauss,
                             update_values | update_gradients | update_JxW_values | update_quadrature_points );

    const unsigned int dofs_per_cell =   fe.dofs_per_cell;
    const unsigned int n_q_points    =   qGauss.size();
    
    std::vector<double>         phiAtQ(     n_q_points );
    std::vector<Tensor<1,dim>>  phiGradAtQ( n_q_points, Tensor<1,dim>() );

    FullMatrix<double> cell_matrix( dofs_per_cell, dofs_per_cell);
    Vector<double>        cell_rhs( dofs_per_cell );
    
    const double time_step = time_iterator.get_current_time_increment();

    for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      std::vector<types::global_dof_index> local_dof_indices( dofs_per_cell );
      
      cell_matrix = 0;
      cell_rhs    = 0;

      fe_values.reinit(cell);
      fe_values.get_function_values(     solution_levelset, phiAtQ ); // compute values of old solution
      fe_values.get_function_gradients(  solution_levelset, phiGradAtQ ); // compute values of old solution

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
        auto qCoord = fe_values.get_quadrature_points()[q_index];
        const Tensor<1, dim> a =  field_conditions->advection_field->value( qCoord );


        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<dofs_per_cell; ++j)
          {
              auto velocity_grad_phi_j = a * fe_values.shape_grad( j, q_index);  // grad_phi_j(x_q)
              cell_matrix( i, j ) += (  fe_values.shape_value( i, q_index) * 
                                       fe_values.shape_value( j, q_index) +
                                       parameters.theta * time_step * ( parameters.artificial_diffusivity * 
                                                             fe_values.shape_grad( i, q_index) * 
                                                             fe_values.shape_grad( j, q_index) +
                                                             fe_values.shape_value( i, q_index) ) *
                                                             velocity_grad_phi_j 
                                    ) * fe_values.JxW(q_index);                                    
          }

          cell_rhs( i ) +=
            (  fe_values.shape_value( i, q_index) * phiAtQ[q_index]
                - 
               ( 1. - parameters.theta ) * time_step * 
                 (
                   parameters.artificial_diffusivity *
                   fe_values.shape_grad( i, q_index) *
                   phiGradAtQ[q_index]
                   +
                   a * phiGradAtQ[q_index]    
                     * fe_values.shape_value(  i, q_index)
                 )
              ) * fe_values.JxW(q_index) ;      
        }
      }// end gauss
    
      // assembly
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix,
                                             cell_rhs,
                                             local_dof_indices,
                                             system_matrix,
                                             system_rhs);
       
    } // end element loop

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
    
    LinearSolve<VectorType>::solve( system_matrix,
                                    solution_levelset,
                                    system_rhs,
                                    mpi_communicator);
    constraints.distribute(solution_levelset);
    solution_levelset.update_ghost_values();
  }


  template <int dim, int degree>
  void 
  LevelSetEquation<dim,degree>::initialize_reinitialization_model()
  {
    ReinitializationData reinit_data;
    reinit_data.reinit_model        = ReinitModelType::olsson2007;
    reinit_data.d_tau               = GridTools::minimal_cell_diameter(triangulation);
    reinit_data.degree              = parameters.levelset_degree;
    reinit_data.verbosity_level     = utilityFunctions::VerbosityType::major;
    reinit_data.min_cell_size       = GridTools::minimal_cell_diameter(triangulation);
    reinit_data.do_print_l2norm     = parameters.output_norm_levelset;
    reinit_data.do_matrix_free      = parameters.do_matrix_free;
    
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
  void 
  LevelSetEquation<dim,degree>::compute_reinitialization_model()
  {
    reini.solve( solution_levelset );
  }

  template <int dim, int degree>
  void 
  LevelSetEquation<dim,degree>::initialize_curvature()
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
  void 
  LevelSetEquation<dim,degree>::compute_curvature()
  {
    curvature.solve( solution_levelset );
  }
  
  template <int dim, int degree>
  void
  LevelSetEquation<dim,degree>::initialize_time_iterator()
  {
    TimeIteratorData time_data;
    time_data.start_time       = 0.0;
    time_data.end_time         = parameters.end_time;
    time_data.time_increment   = parameters.time_step_size; 
    time_data.max_n_time_steps = 1E10; // this criteria is set to be not relevant for the level set problem
    
    time_iterator.initialize(time_data);
  }
 
  template <int dim, int degree>
  void 
  LevelSetEquation<dim,degree>::output_results( )
  {
    const double time_step = time_iterator.get_current_time_step_number();
    if (parameters.compute_paraview_output)
    {
        TimerOutput::Scope t(computing_timer, "output_results");   

        std::vector<std::string> solution_names(dim, "velocity");

        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);

        utilityFunctions::GradientPostprocessor<dim>    gradient_postprocessor;

        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution_levelset,                      "phi");

        data_out.add_data_vector(solution_levelset, gradient_postprocessor);
        
        //data_out.add_data_vector(normal_vector_field.block(0), "normal_x");
        //data_out.add_data_vector(normal_vector_field.block(1), "normal_y");
        //data_out.add_data_vector(curvature_field,              "curvature");
         Vector<float> subdomain(triangulation.n_active_cells());
        for (unsigned int i = 0; i < subdomain.size(); ++i)
          subdomain(i) = triangulation.locally_owned_subdomain();
        data_out.add_data_vector(subdomain, "subdomain");
     
        data_out.build_patches();

        data_out.write_vtu_with_pvtu_record(
            "./", parameters.filename_paraview_output, time_step, mpi_communicator, 2, 8);
    }
    //if (parameters.compute_volume_output)
        //compute_overall_phase_volume();
  }
  
  
  template <int dim, int degree>
  void 
  LevelSetEquation<dim,degree>::run( )
  {
    timer.start();
    pcout << "Running "
          << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    initialize_levelset();

    if ( parameters.activate_reinitialization )    
        initialize_reinitialization_model();
    
    output_results(); // print initial state

    //const int p = parameters.levelset_degree;
    //const double CFL = 0.2/(p*p);
    //const double dx = GridTools::minimal_cell_diameter(triangulation) / std::sqrt(dim);

    initialize_time_iterator(); 

    while ( !time_iterator.is_finished() )
    {
        const double dt = time_iterator.get_next_time_increment();
        pcout << "Time step " << time_iterator.get_current_time_step_number() << " at current t=" << time_iterator.get_current_time() << std::endl;

        compute_levelset_model(); // @todo: insert updateFlag

        if (parameters.output_norm_levelset)
            pcout << " (not reinitialized) levelset function ||phi|| = " << std::setprecision(10) << solution_levelset.l2_norm() << std::endl;

        if ( parameters.activate_reinitialization )    
            compute_reinitialization_model();

        if (parameters.output_norm_levelset)
            pcout << " (reinitialized) levelset function ||phi|| = " << std::setprecision(10) << solution_levelset.l2_norm() << std::endl;

        output_results();

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
    }
  }
  
  // instantiation
  template class LevelSetEquation<2,1>; 
  template class LevelSetEquation<2,2>; 

} // end of namespace LevelSetParallel

/*
 *@todo: shift the following functions to a postprocess class
 *
  template <int dim, int degree>
  void LevelSetEquation<dim,degree>::compute_error( const Function<dim>& ExactSolution )
  {
    const auto qGauss = QGauss<dim>(parameters.levelset_degree+1);
    Vector<double> norm_per_cell(triangulation.n_active_cells());

    VectorTools::integrate_difference(dof_handler,
                                      solution_levelset,
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
                                      solution_levelset,
                                      ExactSolution,
                                      difference_per_cell,
                                      qGauss,
                                      VectorTools::L1_norm);

    double h1_error = VectorTools::compute_global_error(triangulation,
                                                        difference_per_cell,
                                                        VectorTools::L1_norm);
    pcout << "L1 error = " << h1_error << std::endl;
    
  }

  //// @ to be rearranged
  template <int dim, int degree>
  void LevelSetEquation<dim,degree>::compute_overall_phase_volume( )
  {
    Vector<double> phase_volume_per_cell(triangulation.n_active_cells());

    const auto qGauss = QGauss<dim>(parameters.levelset_degree+1);
    
    FEValues<dim> fe_values( fe,
                             qGauss,
                             update_values | update_JxW_values | update_quadrature_points );

    const unsigned int dofs_per_cell =   fe.dofs_per_cell;
    
    const unsigned int n_q_points    = qGauss.size();
    std::vector<double> phi_at_q(  n_q_points );

    //const double& max_value = solution_levelset.max();
    //const double& min_value = solution_levelset.min();
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
        fe_values.get_function_values(  solution_levelset, phi_at_q ); // compute values of old solution

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
        //if ( time==parameters.start_time )
        //{
            //std::cout << "output file opened" << std::endl;
            //std::fstream fs;
            //fs.open (parameters.filename_volume_output, std::fstream::out);
            //fs.precision(10);
            //fs << "time | volume phase 1 | volume phase 2 " << std::endl; 
            //fs << std::left << std::setw(headerWidths[0]) << time;
            //fs << "   " << std::left << std::setw(headerWidths[1]) << volume_fraction[0]; 
            //fs << "   " << std::left << std::setw(headerWidths[1]) << volume_fraction[1] << std::endl; 
            //fs.close();
        //}
        //else
        //{
            //std::fstream fs;
            //fs.open (parameters.filename_volume_output,std::fstream::in | std::fstream::out | std::fstream::app);
            //fs.precision(10);
            //fs << std::left << std::setw(headerWidths[0]) << time;
            //fs << "   " << std::left << std::setw(headerWidths[1]) << volume_fraction[0]; 
            //fs << "   " << std::left << std::setw(headerWidths[1]) << volume_fraction[1] << std::endl; 
        //}
    }
  }

 */

