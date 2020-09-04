/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, August 2020
 *
 * ---------------------------------------------------------------------*/

// to access minimal_cell_diamater
#include <deal.II/grid/grid_tools.h>
// to use DoFTools::
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
// for matrix free solution
#include <deal.II/fe/mapping_q.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/lac/solver_control.h> // for reduction_control
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/vector_operation.h>
#include <deal.II/lac/trilinos_solver.h>
// for output
#include <deal.II/numerics/data_out.h>

// MeltPoolDG
#include "utilityfunctions.hpp"
#include "normalvectoroperator.hpp"
#include "linearsolve.hpp"
#include "reinitialization.hpp"
#include "timeiterator.hpp"
#include "reinitializationoperator.hpp"

namespace MeltPoolDG
{
  using namespace dealii; 

  template <int dim, int degree>
  Reinitialization<dim,degree>::Reinitialization(std::shared_ptr<SimulationBase<dim>> base_in )
  : mpi_communicator(    base_in->get_mpi_communicator())
  , pcout(               std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  , normal_vector_field( mpi_communicator )
  , module_dof_handler(  base_in->triangulation )
  //, triangulation(       base_in->triangulation )
  , field_conditions(    base_in->get_field_conditions()  )
  , min_cell_size(       GridTools::minimal_cell_diameter(base_in->triangulation) )
  {
    initialize_data_from_global_parameters(base_in->parameters);
    initialize_module( base_in );

    // the distributed sparsity pattern is only there to fill the system matrix once
    TrilinosWrappers::SparsityPattern dsp( this->locally_owned_dofs,
                                           this->locally_owned_dofs,
                                           this->locally_relevant_dofs,
                                           mpi_communicator);

    DoFTools::make_sparsity_pattern(this->module_dof_handler, 
                                    dsp,
                                    this->module_constraints,
                                     Utilities::MPI::this_mpi_process(mpi_communicator)
                                    );
    dsp.compress();

    // submodule
    initialize( this->reinit_data,
                dsp, 
                this->module_dof_handler,
                this->module_constraints,
                this->locally_owned_dofs,
                this->locally_relevant_dofs
                );
  }

  template <int dim, int degree>
  void
  Reinitialization<dim,degree>::initialize_module(std::shared_ptr<SimulationBase<dim>> base_in)
  {
    module_dof_handler.distribute_dofs( FE_Q<dim>(degree) );
    
    locally_owned_dofs = module_dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(module_dof_handler, locally_relevant_dofs);
    
    module_constraints.clear();
    module_constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(module_dof_handler, module_constraints);
    module_constraints.close();     

  }

  template <int dim, int degree>
  void
  Reinitialization<dim,degree>::initialize_data_from_global_parameters(const LevelSetParameters& data_in)
  {
    //@ todo: add parameter for paraview output
    reinit_data.reinit_model        = static_cast<ReinitModelType>(data_in.reinit_modeltype);
    reinit_data.d_tau               = data_in.reinit_dtau > 0.0 ? 
                                      data_in.reinit_dtau
                                      : min_cell_size;
    reinit_data.constant_epsilon    = data_in.reinit_constant_epsilon;
    reinit_data.max_reinit_steps    = data_in.reinit_max_n_steps; //parameters.max_reinitializationsteps;
    //reinit_data.verbosity_level     = utilityFunctions::VerbosityType::major;
    reinit_data.do_print_l2norm     = data_in.reinit_do_print_l2norm; //parameters.output_norm_levelset;
    reinit_data.do_matrix_free      = data_in.reinit_do_matrixfree;
  }

  /*
   *  this constructor is only called when renitialization is used as a submodule
   */
  template <int dim, int degree>
  Reinitialization<dim,degree>::Reinitialization(const MPI_Comm & mpi_communicator_in)
  : mpi_communicator(     mpi_communicator_in )
  , pcout(                std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  , normal_vector_field(  mpi_communicator_in )
  {
  }

  template <int dim, int degree>
  void
  Reinitialization<dim,degree>::initialize( const ReinitializationData&           data_in,
                                            const SparsityPatternType&            dsp_in,
                                            const DoFHandlerType&                 dof_handler_in,
                                            const ConstraintsType&                constraints_in,
                                            const IndexSet&                       locally_owned_dofs_in,
                                            const IndexSet&                       locally_relevant_dofs_in
                                          )
  {
      reinit_data           = data_in;
      dof_handler           = &dof_handler_in;
      constraints           = &constraints_in; //@ this is a hack!!!
      locally_owned_dofs    = locally_owned_dofs_in;
      locally_relevant_dofs = locally_relevant_dofs_in;
      
      // @todo: for matrix only
      system_matrix.reinit( dsp_in );

      system_rhs.reinit( locally_owned_dofs, 
                         locally_relevant_dofs,
                         mpi_communicator ); 
      
      //const bool verbosity_active = ((Utilities::MPI::this_mpi_process(mpi_commun) == 0) && (reinit_data.verbosity_level!=utilityFunctions::VerbosityType::silent));
      //this->pcout.set_condition(verbosity_active);

      /*
       * initialize the normal_vector_field computation
       * @ todo: how should data be transferred from the base class
       */
      
      solution_normal_vector.reinit( dim ); 
      
      for (unsigned int d=0; d<dim; ++d)
          solution_normal_vector.block(d).reinit(locally_owned_dofs,
                                                 locally_relevant_dofs,
                                                 mpi_communicator);
      
      //@ introduce new C++20 features --> shift to normalvector class
      NormalVectorData normal_vector_data;
      normal_vector_data.damping_parameter = min_cell_size * 0.5;

      normal_vector_data.verbosity_level   = reinit_data.verbosity_level;
      //normal_vector_data.min_cell_size     = reinit_data.min_cell_size;
      normal_vector_data.do_print_l2norm   = reinit_data.do_print_l2norm;

      normal_vector_field.initialize( normal_vector_data, 
                                      dsp_in,
                                      dof_handler_in,
                                      constraints_in,
                                      locally_owned_dofs_in,
                                      locally_relevant_dofs,
                                      min_cell_size);
  }

  template <int dim, int degree>
  void 
  Reinitialization<dim,degree>::solve( VectorType & solution_out )
  {
      switch(reinit_data.reinit_model)
      {
          case ReinitModelType::olsson2007:
              if (reinit_data.do_matrix_free)
                solve_olsson_model_matrixfree( solution_out );
              else
                solve_olsson_model( solution_out );
              break;
          default:
              AssertThrow(false, ExcMessage("Requested reinitialization model not implemented."))
              break;
      }
  }

  template <int dim, int degree>
  void 
  Reinitialization<dim,degree>::solve_olsson_model_matrixfree( VectorType & solution_out )
  {
    pcout << " >>>>>>>>>>>>>>>>>>> REINITIALIZATION START (matrix-free)" << std::endl;
    normal_vector_field.compute_normal_vector_field_matrixfree( solution_out, solution_normal_vector );
    solution_normal_vector.update_ghost_values();
    
    MappingQ<dim> mapping(degree);
    QGauss<1>     quad_1d(degree + 1);
    
    typedef VectorizedArray<double>  VectorizedArrayType;
    typename MatrixFree<dim, double, VectorizedArrayType>::AdditionalData  additional_data;
    typedef LevelSetMatrixFree::ReinitializationOperator<dim, degree> OperatorType; 
    additional_data.mapping_update_flags = update_values | update_gradients;

    MatrixFree<dim, double, VectorizedArrayType> matrix_free;
    matrix_free.reinit(mapping, *dof_handler, *constraints, quad_1d, additional_data);

    OperatorType rei( matrix_free,
                      reinit_data.d_tau,
                      min_cell_size / ( std::sqrt(2)*2. ));

    VectorType src, rhs, solution;
    BlockVectorType normals(dim); 

    rei.initialize_dof_vector(src);
    rei.initialize_dof_vector(rhs);
    rei.initialize_dof_vector(solution);
    
    rei.set_normal_vector_field(solution_normal_vector);
    
    // @todo: move to create rhs;
    solution.copy_locally_owned_data_from(solution_out);
    solution.update_ghost_values();

    std::shared_ptr<TimeIterator> time_iterator = std::make_shared<TimeIterator>();
    initialize_time_iterator(time_iterator); 

    table.clear();
    while ( !time_iterator->is_finished() )
    {
        const double d_tau = time_iterator->get_next_time_increment();  
        rei.set_time_increment(d_tau);
        
        // create right hand side
        matrix_free.initialize_dof_vector(rhs);
        rei.create_rhs(rhs, solution,solution_normal_vector);
        
        matrix_free.initialize_dof_vector(src);
        
        // @ todo: how to introduce preconditioner for matrix-free solution?
        const int iter = LinearSolve<VectorType,SolverCG<VectorType>, OperatorType>::solve( rei,
                                                                           src,
                                                                           rhs);
        constraints->distribute(src);

        solution += src;
        solution.update_ghost_values();
        

        pcout << "   with " << iter << " GMRES iterations.";
        
        pcout << "\t\t |ΔΨ|∞ = " << std::setprecision(10) << src.linfty_norm();
        pcout << " |ΔΨ|²/dT = " << std::setprecision(10) << src.l2_norm()/d_tau << std::endl;
        
        table.add_value("t",       time_iterator->get_current_time());
        table.add_value("iter ",    iter);
        table.add_value("|ΔΨ|∞",    src.linfty_norm());
        table.add_value("|ΔΨ|²/dT", src.l2_norm()/d_tau);
    }
    table.set_precision("t",        6);
    table.set_precision("|ΔΨ|∞",    10);
    table.set_precision("|ΔΨ|²/dT", 10);

    //if( Utilities::MPI::this_mpi_process(mpi_communicator) == 0) 
      //table.write_text(std::cout); 
    
    solution_out = solution;
    solution_out.update_ghost_values();
    pcout << "       >>>>>>>>>>>>>>>>>>> REINITIALIZATION END " << std::endl;
  }

  template <int dim, int degree>
  void 
  Reinitialization<dim,degree>::solve_olsson_model( VectorType & solution_out )
  {
      pcout << " >>>>>>>>>>>>>>>>>>> REINITIALIZATION START" << std::endl;
      VectorType solution_in; //solution_out;
      solution_in.reinit(locally_owned_dofs,
                         locally_relevant_dofs,
                         mpi_communicator);
      solution_in.copy_locally_owned_data_from(solution_out);
      solution_in.update_ghost_values();

      
      auto qGauss = QGauss<dim>( degree+1 );
      
      FE_Q<dim> fe( degree );
      FEValues<dim> fe_values( fe,
                               qGauss,
                               update_values | update_gradients | update_quadrature_points | update_JxW_values );

      const unsigned int dofs_per_cell = fe.dofs_per_cell;
      
      FullMatrix<double>   cell_matrix( dofs_per_cell, dofs_per_cell );
      Vector<double>       cell_rhs(    dofs_per_cell );
      
      const unsigned int n_q_points    = qGauss.size();

      std::vector<double>         psiAtQ(     n_q_points );
      std::vector<Tensor<1,dim>>  normal_at_quadrature(  n_q_points, Tensor<1,dim>() );
      std::vector<Tensor<1,dim>>  psiGradAtQ( n_q_points, Tensor<1,dim>() );
      
      std::vector<types::global_dof_index> local_dof_indices( dofs_per_cell );
      /*
       * compute vector field of normals to the current solution of the level set function
       * @todo: is there an option that normal vectors are also accessible from the base level set class 
       *        for output routines?
       */
      
      normal_vector_field.compute_normal_vector_field( solution_in, solution_normal_vector );
      solution_normal_vector.update_ghost_values();
      
      std::shared_ptr<TimeIterator> time_iterator = std::make_shared<TimeIterator>();

      initialize_time_iterator(time_iterator); 
      
      table.clear();
      while ( !time_iterator->is_finished() )
      {
        const double d_tau = time_iterator->get_next_time_increment();
        system_rhs      = 0.0;
        system_matrix   = 0.0;
        
        for (const auto &cell : dof_handler->active_cell_iterators())
        if (cell->is_locally_owned())
        {
           cell_matrix = 0.0;
           cell_rhs    = 0.0;
           fe_values.reinit(cell);
           
           const double epsilon_cell = reinit_data.constant_epsilon>0.0 ? reinit_data.constant_epsilon : cell->diameter() / ( std::sqrt(dim) * 2 );
           AssertThrow(epsilon_cell>0.0, ExcMessage("Reinitialization: the value of epsilon for the reinitialization function must be larger than zero!"));

           fe_values.get_function_values(     solution_out, psiAtQ );     // compute values of old solution at tau_n
           fe_values.get_function_gradients(  solution_out, psiGradAtQ ); // compute gradients of old solution at tau_n
            
           normal_vector_field.get_unit_normals_at_quadrature(fe_values,
                                                              solution_normal_vector,
                                                              normal_at_quadrature);

           // @todo: only compute normals once during timestepping
           for (const unsigned int q_index : fe_values.quadrature_point_indices())
            {
               for (const unsigned int i : fe_values.dof_indices())
               {
                   const double nTimesGradient_i = normal_at_quadrature[q_index] * fe_values.shape_grad(i, q_index);

                   for (const unsigned int j : fe_values.dof_indices())
                   {
                       const double nTimesGradient_j = normal_at_quadrature[q_index] * fe_values.shape_grad(j, q_index);
                       cell_matrix(i,j) += (
                                             fe_values.shape_value(i,q_index) * fe_values.shape_value(j,q_index)
                                             + 
                                             d_tau * epsilon_cell * nTimesGradient_i * nTimesGradient_j
                                           ) 
                                           * 
                                           fe_values.JxW( q_index );
                   }
                  
                   const double diffRhs = epsilon_cell * normal_at_quadrature[q_index] * psiGradAtQ[q_index];

                   cell_rhs(i) += ( 0.5 * ( 1. - psiAtQ[q_index] * psiAtQ[q_index] ) - diffRhs )
                                   *
                                   nTimesGradient_i 
                                   *
                                   d_tau 
                                   * 
                                   fe_values.JxW( q_index );
                   
               }                                    
          }// end loop over gauss points
          
          // assembly
          cell->get_dof_indices( local_dof_indices );
          
          constraints->distribute_local_to_global( cell_matrix,
                                                   cell_rhs,
                                                   local_dof_indices,
                                                   system_matrix,
                                                   system_rhs);
           
        }
        system_matrix.compress( VectorOperation::add );
        system_rhs.compress(    VectorOperation::add );

        // @ here is space for improvement
        VectorType    re_solution_u_temp( locally_owned_dofs,
                                           mpi_communicator );
        VectorType    re_delta_solution_u(locally_owned_dofs,
                                           mpi_communicator );
        re_solution_u_temp = solution_out;
        
        /* 
         *  iterative solver
         */
        TrilinosWrappers::PreconditionAMG preconditioner;     
        TrilinosWrappers::PreconditionAMG::AdditionalData data;     
        preconditioner.initialize(system_matrix, data); 

        const int iter = LinearSolve<VectorType,
                                     SolverGMRES<VectorType>,
                                     SparseMatrixType,
                                     TrilinosWrappers::PreconditionAMG>::solve( system_matrix,
                                                                                re_delta_solution_u,
                                                                                system_rhs,
                                                                                preconditioner);
        constraints->distribute( re_delta_solution_u );

        re_solution_u_temp += re_delta_solution_u;
        
        solution_out = re_solution_u_temp;
        solution_out.update_ghost_values();

        output_results(solution_out, time_iterator->get_current_time_step_number());
        
        if(reinit_data.do_print_l2norm)
        {
            pcout << "   with " << iter << " GMRES iterations.";
            pcout << "\t\t |ΔΨ|∞ = " << std::setprecision(10) << re_delta_solution_u.linfty_norm();
            pcout << " |ΔΨ|²/dT = " << std::setprecision(10) << re_delta_solution_u.l2_norm()/d_tau << std::endl;
        }
        table.add_value("t",       time_iterator->get_current_time());
        table.add_value("iter ",    iter);
        table.add_value("|ΔΨ|∞",    re_delta_solution_u.linfty_norm());
        table.add_value("|ΔΨ|²/dT", re_delta_solution_u.l2_norm()/d_tau);
      }
      table.set_precision("t",        6);
      table.set_precision("|ΔΨ|∞",    10);
      table.set_precision("|ΔΨ|²/dT", 10);

      //if( Utilities::MPI::this_mpi_process(mpi_communicator) == 0) 
        //table.write_text(std::cout); 

      pcout << "       >>>>>>>>>>>>>>>>>>> REINITIALIZATION END " << std::endl;
  }
  
  template <int dim, int degree>
  void
  Reinitialization<dim,degree>::run( )
  {
    pcout << "Number of degrees of freedom: " << dof_handler->n_dofs()           
                                 << std::endl << std::endl;
    
    VectorType solution_ls;
    solution_ls.reinit( locally_owned_dofs, 
                        locally_relevant_dofs,
                        mpi_communicator);

    VectorTools::project( module_dof_handler, 
                          module_constraints,
                          QGauss<dim>(degree+1),
                          *field_conditions->initial_field,           
                          solution_ls );
    solution_ls.update_ghost_values();
    
    solve(solution_ls);

  }

  template <int dim, int degree>
  void 
  Reinitialization<dim,degree>::output_results(const VectorType& solution_in,
                                               const double      time)
  {
    //if (parameters.paraview_do_output)
    //{
      DataOut<dim> data_out;
      data_out.attach_dof_handler(*dof_handler);
      data_out.add_data_vector(solution_in, "phi");
      
      /*
      BlockVectorType normal_vector_field;
      normal_vector_field.reinit(dim);
      for (int d=0; d<dim; ++d)
        normal_vector_field.block(d).reinit(locally_owned_dofs,
                          locally_relevant_dofs,
                          mpi_communicator);

      if (parameters.paraview_print_normal_vector)
      {
        normal_vector_field = reini.get_normal_vector_field();
        for (int d=0; d<dim; d++)
        {
          std::string name = "n_"+std::to_string(d);
          data_out.add_data_vector(dof_handler, normal_vector_field.block(d), name);
        }
       }
       */
      
      VectorType levelset_exact;
      levelset_exact.reinit( locally_owned_dofs,
                              mpi_communicator);
      /*
      if (parameters.paraview_print_exactsolution)
      {
          VectorTools::project( dof_handler, 
                                constraints_no_dirichlet,
                                QGauss<dim>(degree+1),
                                *field_conditions->exact_solution_field,           
                                levelset_exact);
          data_out.add_data_vector(dof_handler, levelset_exact, "exactsolution");
      }
      */

      //Vector<float> subdomain(triangulation.n_active_cells());
      //for (unsigned int i = 0; i < subdomain.size(); ++i)
        //subdomain(i) = triangulation.locally_owned_subdomain();
      //data_out.add_data_vector(subdomain, "subdomain");
      
      data_out.build_patches();
      
      const int n_digits_timestep = 2;
      const int n_groups = 1;
      data_out.write_vtu_with_pvtu_record("./", "solution_reinitialization", time, mpi_communicator, n_digits_timestep, n_groups);
  }
  
  
  template <int dim, int degree>
  void
  Reinitialization<dim,degree>::initialize_time_iterator( std::shared_ptr<TimeIterator> t )
  {
      // @ shift into own function ?
      TimeIteratorData time_data;
      time_data.start_time       = 0.0;
      time_data.end_time         = 100.;
      time_data.time_increment   = reinit_data.d_tau; 
      time_data.max_n_time_steps = reinit_data.max_reinit_steps;
      
      t->initialize(time_data);
  }

  template <int dim, int degree>
  void
  Reinitialization<dim,degree>::print_me( )
  {
      pcout << "hello from reinitialization"                                  << std::endl;   
      // @ is there a more elegant solution?
      pcout << "reinit_model: "               << static_cast<std::underlying_type<ReinitModelType>::type>(reinit_data.reinit_model) << std::endl;
      pcout << "d_tau: "                      << reinit_data.d_tau            << std::endl;
      pcout << "constant_epsilon: "           << reinit_data.constant_epsilon << std::endl;
      pcout << "max reinit steps: "           << reinit_data.max_reinit_steps << std::endl;
  }
  
  template <int dim, int degree>
  LinearAlgebra::distributed::BlockVector<double> 
  Reinitialization<dim,degree>::get_normal_vector_field() const
  {
    return solution_normal_vector; 
  }

  // instantiation
  template class Reinitialization<2,1>;
  template class Reinitialization<2,2>;
  //template class Reinitialization<3>; // temporarily disabled to work on matrixfree implementation
} // namespace MeltPoolDG


