/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, August 2020
 *
 * ---------------------------------------------------------------------*/

// to access minimal_cell_diamater
#include <deal.II/grid/grid_tools.h>
// to use preconditioner 
#include <deal.II/lac/petsc_precondition.h>
// to use DoFTools::
#include <deal.II/dofs/dof_tools.h>

#include <reinitialization.hpp>

#include <timeiterator.hpp>

namespace LevelSetParallel
{
    using namespace dealii; 

    template <int dim>
    Reinitialization<dim>::Reinitialization(const MPI_Comm & mpi_commun_in)
    : mpi_commun( mpi_commun_in )
    , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_commun) == 0)
    , normal_vector_field( mpi_commun_in )
    {
    }

    template <int dim>
    void
    Reinitialization<dim>::initialize( const ReinitializationData &  data_in,
                                       const SparsityPatternType&    dsp_in,
                                       const DoFHandler<dim> &       dof_handler_in,
                                       const ConstraintsType&        constraints_in,
                                       const IndexSet&               locally_owned_dofs_in,
                                       const IndexSet&               locally_relevant_dofs_in
                                    )
    {
        reinit_data           = data_in;
        dof_handler           = &dof_handler_in;
        constraints           = &constraints_in;
        locally_owned_dofs    = locally_owned_dofs_in;
        locally_relevant_dofs = locally_relevant_dofs_in;
        
        system_matrix.reinit( locally_owned_dofs,
                              locally_owned_dofs,
                              dsp_in,
                              mpi_commun );

        system_rhs.reinit( locally_owned_dofs, 
                           mpi_commun ); 
        
        const bool verbosity_active = ((Utilities::MPI::this_mpi_process(mpi_commun) == 0) && (reinit_data.verbosity_level!=utilityFunctions::VerbosityType::silent));
        this->pcout.set_condition(verbosity_active);

        /*
         * initialize the normal_vector_field computation
         * @ todo: how should data be transferred from the base class
         */
        
        solution_normal_vector.reinit( dim ); 
        
        for (unsigned int d=0; d<dim; ++d)
            solution_normal_vector.block(d).reinit(locally_owned_dofs,
                                                   locally_relevant_dofs,
                                                   mpi_commun);
        
        //@ introduce new C++20 features
        NormalVectorData normal_vector_data;
        normal_vector_data.damping_parameter = 1e-6;
        normal_vector_data.degree            = reinit_data.degree;
        normal_vector_data.verbosity_level   = reinit_data.verbosity_level;

        normal_vector_field.initialize( normal_vector_data, 
                                        dsp_in,
                                        dof_handler_in,
                                        constraints_in,
                                        locally_owned_dofs_in);
    }

    template <int dim>
    void 
    Reinitialization<dim>::solve( VectorType & solution_out )
    {
        switch(reinit_data.reinit_model)
        {
            case ReinitModelType::olsson2007:
                solve_olsson_model( solution_out );
                break;
            default:
                AssertThrow(false, ExcMessage("Requested reinitialization model not implemented."))
                break;
        }
    }

    template <int dim>
    void 
    Reinitialization<dim>::solve_olsson_model( VectorType & solution_out )
    {
        pcout << "       >>>>>>>>>>>>>>>>>>> REINITIALIZATION START " << std::endl;

        VectorType solution_in = solution_out;
        
        auto qGauss = QGauss<dim>( reinit_data.degree+1 );
        
        FE_Q<dim> fe( reinit_data.degree );
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
        
        std::shared_ptr<TimeIterator> time_iterator = std::make_shared<TimeIterator>();

        initialize_time_iterator(time_iterator); 
        
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
               
               const double epsilon_cell = ( reinit_data.constant_epsilon>0.0 ) ? reinit_data.constant_epsilon : cell->diameter() / ( std::sqrt(dim) * 2 );
               
               fe_values.get_function_values(     solution_out, psiAtQ );     // compute values of old solution at tau_n
               fe_values.get_function_gradients(  solution_out, psiGradAtQ ); // compute values of old solution at tau_n
                
               normal_vector_field.get_unit_normals_at_quadrature(fe_values,
                                                                  solution_normal_vector,
                                                                  normal_at_quadrature);

               // @todo: only compute normals once during timestepping
               for (const unsigned int q_index : fe_values.quadrature_point_indices())
                {
                   const double diffRhs = epsilon_cell * normal_at_quadrature[q_index] * psiGradAtQ[q_index];

                   for (const unsigned int i : fe_values.dof_indices())
                   {
                       //if (!normalsComputed)
                       //{
                           const double nTimesGradient_i = normal_at_quadrature[q_index] * fe_values.shape_grad(i, q_index);

                           for (const unsigned int j : fe_values.dof_indices())
                           {
                               const double nTimesGradient_j = normal_at_quadrature[q_index] * fe_values.shape_grad(j, q_index);
                               cell_matrix(i,j) += (
                                                     fe_values.shape_value(i,q_index) * fe_values.shape_value(j,q_index)
                                                     + 
                                                     d_tau * epsilon_cell * nTimesGradient_i * nTimesGradient_j
                                                   ) 
                                                   * fe_values.JxW( q_index );
                           }
                       //}

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

            SolverControl solver_control( dof_handler->n_dofs() , 1e-6 * system_rhs.l2_norm() );
            
            LA::SolverCG solver( solver_control, mpi_commun );

            LA::MPI::PreconditionAMG preconditioner;
            LA::MPI::PreconditionAMG::AdditionalData data;
            preconditioner.initialize(system_matrix, data);
            
            // @ here is space for iimprovementn
            VectorType    re_solution_u_temp( locally_owned_dofs,
                                               mpi_commun );
            VectorType    re_delta_solution_u(locally_owned_dofs,
                                               mpi_commun );
            re_solution_u_temp = solution_out;
            
            solver.solve( system_matrix, 
                          re_delta_solution_u, 
                          system_rhs, 
                          preconditioner );

            constraints->distribute( re_delta_solution_u );

            re_solution_u_temp += re_delta_solution_u;
            
            solution_out = re_solution_u_temp;
            solution_out.update_ghost_values();

            time_iterator->print_me( pcout.get_stream() );
            
            pcout << "\t |R|∞ = " << re_delta_solution_u.linfty_norm() << "\t |R|²/dT = ";
            pcout << re_delta_solution_u.l2_norm()/d_tau << "   with " << solver_control.last_step() << " CG iterations." << std::endl;

            if (re_delta_solution_u.l2_norm() / d_tau < 1e-6)
               break;
        } // end of time loop

        pcout << "       >>>>>>>>>>>>>>>>>>> REINITIALIZATION END " << std::endl;
    }
    
    
    template <int dim>
    void
    Reinitialization<dim>::initialize_time_iterator( std::shared_ptr<TimeIterator> t )
    {
        // @ shift into own function ?
        TimeIteratorData time_data;
        time_data.start_time       = 0.0;
        time_data.end_time         = 100.;
        time_data.time_increment   = reinit_data.d_tau; 
        time_data.max_n_time_steps = reinit_data.max_reinit_steps;
        
        t->initialize(time_data);
    }

    template <int dim>
    void
    Reinitialization<dim>::print_me( )
    {
        pcout << "hello from reinitialization"                                  << std::endl;   
        // @ is there a more elegant solution?
        pcout << "reinit_model: "               << static_cast<std::underlying_type<ReinitModelType>::type>(reinit_data.reinit_model) << std::endl;
        pcout << "d_tau: "                      << reinit_data.d_tau            << std::endl;
        pcout << "constant_epsilon: "           << reinit_data.constant_epsilon << std::endl;
        pcout << "max reinit steps: "           << reinit_data.max_reinit_steps << std::endl;
    }

    // instantiation
    template class Reinitialization<2>;
    template class Reinitialization<3>;
} // namespace LevelSetParallel


