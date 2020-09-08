/* ---------------------------------------------------------------------
 * Author: Magdalena Schreter, TUM, 2020
 */
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/fe/mapping_q.h>

#include <meltpooldg/levelset/levelset.hpp>
#include <meltpooldg/reinitialization/reinitialization.hpp>
#include <meltpooldg/curvature/curvature.hpp>
#include <meltpooldg/utilities/linearsolve.hpp>
#include <meltpooldg/utilities/postprocessor.hpp>

namespace MeltPoolDG
{
  using namespace dealii; 

  template <int dim, int degree>
  LevelSetEquation<dim,degree>::LevelSetEquation( std::shared_ptr<SimulationBase<dim>> base_in )
    : mpi_communicator(    base_in->get_mpi_communicator())
    , parameters(          base_in->parameters )
    , fe(                  degree)
    , triangulation(       base_in->triangulation )
    , dof_handler(         base_in->triangulation )
    , pcout(               std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator) == 0) )
    , computing_timer(     base_in->mpi_communicator,
                           pcout,
                           TimerOutput::summary,
                           TimerOutput::wall_times)
    , timer(               mpi_communicator)
    , field_conditions(    base_in->get_field_conditions()  )
    , boundary_conditions( base_in->get_boundary_conditions()  )
    , reini(               this->mpi_communicator )
    , curvature(           this->mpi_communicator )
  {}
  
  template <int dim, int degree>
  void LevelSetEquation<dim,degree>::print_me() 
  {  
    pcout << "Number of active cells: "       << triangulation.n_global_active_cells() << std::endl;
    pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()                  << std::endl << std::endl;
  }

  template <int dim, int degree>
  void LevelSetEquation<dim,degree>::initialize_module()
  {
    // @todo: structure timing of different sections
    //TimerOutput::Scope t(computing_timer, "setup");
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
      VectorTools::interpolate_boundary_values( dof_handler,
                                                bc.first,
                                                *bc.second,
                                                constraints );
    }
    constraints.close();   


    // distributed sparsity pattern --> not required for matrix-free solution algorithm 
    SparsityPatternType dsp( locally_owned_dofs,
                             locally_owned_dofs,
                             locally_relevant_dofs,
                             mpi_communicator);

    DoFTools::make_sparsity_pattern( dof_handler,
                                     dsp,
                                     constraints,
                                     true,
                                     Utilities::MPI::this_mpi_process(mpi_communicator)
                                   );
    dsp.compress();

    system_matrix.reinit( dsp ); 

    // the following are no dirichlet constraints for submodules (reinitialization,
    // normal vector, curvature)
    constraints_no_dirichlet.clear();
    constraints_no_dirichlet.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints_no_dirichlet); 
    constraints_no_dirichlet.close();   

    set_initial_conditions();
  }
  

  template <int dim, int degree>
  void LevelSetEquation<dim,degree>::set_initial_conditions()
  {
    // @ is there a better solution to avoid local copy??
    VectorType solutionTemp( locally_owned_dofs, mpi_communicator);

    VectorTools::project( dof_handler, 
                          constraints,
                          QGauss<dim>(degree+1),
                          *field_conditions->initial_field,           
                          solutionTemp );

    solution_levelset = solutionTemp;
    solution_levelset.update_ghost_values();
  } 

  template <int dim, int degree>
  void LevelSetEquation<dim,degree>::compute_levelset_model()
  {
    // @todo: support timer
    //TimerOutput::Scope t(computing_timer, "assembly");   
    field_conditions->advection_field->set_time( time_iterator.get_current_time() );
    
    system_matrix = 0.0;
    system_rhs    = 0.0;
    
    const auto q_gauss = QGauss<dim>(degree+1);
    
    FEValues<dim> fe_values( fe,
                             q_gauss,
                             update_values | update_gradients | update_JxW_values | update_quadrature_points );

    const unsigned int dofs_per_cell =   fe.dofs_per_cell;
    const unsigned int n_q_points    =   q_gauss.size();
    
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
      fe_values.get_function_values(     solution_levelset, phiAtQ ); 
      fe_values.get_function_gradients(  solution_levelset, phiGradAtQ ); 

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
        auto qCoord = fe_values.get_quadrature_points()[q_index];
        const Tensor<1, dim> a =  field_conditions->advection_field->value( qCoord );


        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<dofs_per_cell; ++j)
          {
              auto velocity_grad_phi_j = a * fe_values.shape_grad( j, q_index);  
              // clang-format off
              cell_matrix( i, j ) += (  fe_values.shape_value( i, q_index) * 
                                       fe_values.shape_value( j, q_index) +
                                       parameters.ls_theta * time_step * ( parameters.ls_artificial_diffusivity * 
                                                             fe_values.shape_grad( i, q_index) * 
                                                             fe_values.shape_grad( j, q_index) +
                                                             fe_values.shape_value( i, q_index) ) *
                                                             velocity_grad_phi_j 
                                    ) * fe_values.JxW(q_index);                                    
              // clang-format on
          }

          // clang-format off
          cell_rhs( i ) +=
            (  fe_values.shape_value( i, q_index) * phiAtQ[q_index]
                - 
               ( 1. - parameters.ls_theta ) * time_step * 
                 (
                   parameters.ls_artificial_diffusivity *
                   fe_values.shape_grad( i, q_index) *
                   phiGradAtQ[q_index]
                   +
                   a * phiGradAtQ[q_index]    
                     * fe_values.shape_value(  i, q_index)
                 )
              ) * fe_values.JxW(q_index) ;      
          // clang-format on
        }
      } // end gauss
    
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
    
    TrilinosWrappers::PreconditionAMG preconditioner;     
    TrilinosWrappers::PreconditionAMG::AdditionalData data;     
    preconditioner.initialize(system_matrix, data); 

    const int iter = LinearSolve<VectorType,
                                 SolverGMRES<VectorType>,
                                 TrilinosWrappers::SparseMatrix,
                                 TrilinosWrappers::PreconditionAMG>::solve( system_matrix,
                                                                            solution_levelset,
                                                                            system_rhs,
                                                                            preconditioner);
    
    pcout << "    GMRES iterations: " << iter;
    constraints.distribute(solution_levelset);
    solution_levelset.update_ghost_values();
  }
  
  // @todo initialize normal vector model
  template <int dim, int degree>
  void 
  LevelSetEquation<dim,degree>::initialize_reinitialization_model()
  {
    ReinitializationData reinit_data;
    reinit_data.reinit_model        = ReinitModelType::olsson2007;
    reinit_data.d_tau               = parameters.reinit_dtau > 0.0 ? 
                                      parameters.reinit_dtau :
                                      GridTools::minimal_cell_diameter(triangulation);
    reinit_data.constant_epsilon    = parameters.reinit_constant_epsilon;
    reinit_data.max_reinit_steps    = parameters.reinit_max_n_steps;
    reinit_data.do_print_l2norm     = parameters.reinit_do_print_l2norm;
    reinit_data.do_matrix_free      = parameters.reinit_do_matrixfree;
    reinit_data.verbosity_level     = TypeDefs::VerbosityType::major;
    

    SparsityPatternType dsp_re( locally_owned_dofs,
                   locally_owned_dofs,
                   locally_relevant_dofs,
                   mpi_communicator);
    DoFTools::make_sparsity_pattern( dof_handler,
                                     dsp_re,
                                     constraints_no_dirichlet,
                                     true,
                                     Utilities::MPI::this_mpi_process(mpi_communicator)
                                   );
    dsp_re.compress();

    reini.initialize_as_submodule( reinit_data , 
                      dsp_re,
                      dof_handler,
                      constraints_no_dirichlet,
                      locally_owned_dofs,
                      locally_relevant_dofs,
                      GridTools::minimal_cell_diameter(triangulation));
  }

  template <int dim, int degree>
  void 
  LevelSetEquation<dim,degree>::compute_reinitialization_model()
  {
    reini.run_as_submodule( solution_levelset );
  }

  template <int dim, int degree>
  void 
  LevelSetEquation<dim,degree>::initialize_curvature()
  {
    CurvatureData curvature_data;
    curvature_data.damping_parameter   = 0.0; // according to the paper by Zahedi (2012)
    curvature_data.min_cell_size       = GridTools::minimal_cell_diameter(triangulation)/std::sqrt(dim);
    curvature_data.verbosity_level     = TypeDefs::VerbosityType::major;
    
    TrilinosWrappers::SparsityPattern dsp_re( locally_owned_dofs,
                                              locally_owned_dofs,
                                              locally_relevant_dofs,
                                              mpi_communicator);
    DoFTools::make_sparsity_pattern( dof_handler,
                                     dsp_re,
                                     constraints_no_dirichlet,
                                     true,
                                     Utilities::MPI::this_mpi_process(mpi_communicator)
                                   );
    dsp_re.compress();
    
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
    time_data.start_time       = 0.0; // @ todo: introduce parameter??
    time_data.end_time         = parameters.ls_end_time;
    time_data.time_increment   = parameters.ls_time_step_size; 
    time_data.max_n_time_steps = 1e9; // this criteria is set to be not relevant for the level set problem
    
    time_iterator.initialize(time_data);
  }
 
  template <int dim, int degree>
  void 
  LevelSetEquation<dim,degree>::output_results(double timestep)
  {
    const double time_step = timestep >= 0.0 ? timestep : time_iterator.get_current_time_step_number();
    if (parameters.paraview_do_output)
    {
      DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(solution_levelset, "phi");
      
      BlockVectorType normal_vector_field;
      normal_vector_field.reinit(dim);
      for (int d=0; d<dim; ++d)
        normal_vector_field.block(d).reinit(locally_owned_dofs,
                          locally_relevant_dofs,
                          mpi_communicator);

      if (parameters.paraview_print_normal_vector)
      {
        std::cout << "normal vectors are prnted: " << std::endl;
        /* currently only supported when reinitialization is activated */
        if ( parameters.ls_do_reinitialization )    
        {
          normal_vector_field = reini.get_normal_vector_field();
          for (int d=0; d<dim; d++)
          {
            std::string name = "n_"+std::to_string(d);
            data_out.add_data_vector(dof_handler, normal_vector_field.block(d), name);
          }
        }
      }
      
      VectorType curvature_field;
      curvature_field.reinit( locally_owned_dofs,
                              locally_relevant_dofs,
                              mpi_communicator);
    
      if (parameters.paraview_print_curvature)
      {
          std::cout << "curvature is plotted " << std::endl;
          initialize_curvature();
          curvature.solve(solution_levelset, curvature_field);
          data_out.add_data_vector(dof_handler, curvature_field, "curvature");
      }
      
      VectorType levelset_exact;
      levelset_exact.reinit( locally_owned_dofs,
                             mpi_communicator);
      
      if (parameters.paraview_print_exactsolution)
      {
        VectorTools::project( dof_handler, 
                              constraints,
                              QGauss<dim>(degree+1),
                              *field_conditions->exact_solution_field,           
                              levelset_exact);

        std::cout << "Norm of exact:" << levelset_exact.l2_norm() << std::endl;
        data_out.add_data_vector(dof_handler, levelset_exact, "exactsolution");
      }

      Vector<float> subdomain(triangulation.n_active_cells());
      for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
      data_out.add_data_vector(subdomain, "subdomain");
      
      data_out.build_patches();
      
      const int n_digits_timestep = 2; //@todo: add to parameters!!
      const int n_groups = 1;          //@todo: add to parameters!!
      data_out.write_vtu_with_pvtu_record(
          "./", parameters.paraview_filename, time_step, mpi_communicator, n_digits_timestep, n_groups);
    }
  }
  
  // @todo: this function is currently a mess :) !
  template <int dim, int degree>
  void 
  LevelSetEquation<dim,degree>::run()
  {
    // @todo: include timer
    // timer.start();
    pcout << "Running "
          << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;
    
    initialize_module();
    
    Postprocessor<dim> postprocessor(mpi_communicator);
    print_me();
    
    if ( parameters.ls_do_reinitialization )    
    {
        initialize_reinitialization_model();
        compute_reinitialization_model();
    }
    output_results(0.0); // print initial state

    initialize_time_iterator(); 

    while ( !time_iterator.is_finished() )
    {
        time_iterator.get_next_time_increment();
        pcout << "Time step " << time_iterator.get_current_time_step_number() << " at current t=" << std::setprecision(10) << time_iterator.get_current_time() << std::endl;

        compute_levelset_model(); // @todo: insert updateFlag

        if (parameters.ls_do_print_l2norm)
            pcout << " levelset function ||phi|| = " << std::setprecision(10) << solution_levelset.l2_norm() << std::endl;

        if ( parameters.ls_do_reinitialization )    
          compute_reinitialization_model();

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
      
      if (parameters.compute_volume_output)
      {
        auto volume_frac = postprocessor.compute_volume_of_phases( degree,
                                  degree+1,
                                  dof_handler,
                                  solution_levelset,
                                  time_iterator.get_current_time(),
                                  mpi_communicator
                                );

        postprocessor.collect_volume_fraction( volume_frac );
      }
    }
    

    if (parameters.compute_volume_output)
      postprocessor.print_volume_fraction_table(mpi_communicator, parameters.filename_volume_output);
    if (parameters.do_compute_error)
      postprocessor.compute_error( degree+1,
                                   solution_levelset,
                                   *field_conditions->exact_solution_field,  
                                    dof_handler,
                                   triangulation         
                                   );
  }
  
  // instantiation
  template class LevelSetEquation<2,1>; 
  template class LevelSetEquation<2,2>; 

} // namespace MeltPoolDG

