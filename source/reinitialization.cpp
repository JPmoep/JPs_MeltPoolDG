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

#include <reinitializationoperator.hpp>
// for direct solver
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/lac/solver_control.h> // for reduction_control
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/vector_tools.h>
#include "utilityFunctions.hpp"

#include <deal.II/lac/vector_operation.h>

namespace LevelSetParallel
{
    template <int dim>
    class InitialValues : public dealii::Function<dim>
    {
    private:
      Point<dim>   point{0.5, 0.5};
      const double rad = 0.1;

    public:
      InitialValues()
        : Function<dim>(1)
      {}

      virtual double
      value(const dealii::Point<dim> &p,
            const unsigned int        component = 0) const override
      {
        (void)component;
        double dist = point.distance(p);
        return 0.5 * (1.0 - std::tanh(2 * (dist - rad)));
      }
    };

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
        normal_vector_data.min_cell_size     = reinit_data.min_cell_size;
        normal_vector_data.do_print_l2norm   = reinit_data.do_print_l2norm;

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
                solve_olsson_model_matrixfree( solution_out );
                solve_olsson_model( solution_out );
                break;
            default:
                AssertThrow(false, ExcMessage("Requested reinitialization model not implemented."))
                break;
        }
    }


    template <int dim>
    void 
    Reinitialization<dim>::solve_olsson_model_matrixfree( VectorType & solution_out )
    {
        normal_vector_field.compute_normal_vector_field( solution_out, solution_normal_vector );
        
        pcout << "----------- MATRIX FREE " << std::endl;
        std::cout << "norm solution out " << solution_out.l2_norm() << std::endl;

        PETScVectorType ls;
        ls.reinit(locally_owned_dofs,
                  locally_relevant_dofs,
                  mpi_commun);
       
        ls = solution_out;

        const unsigned int dimTemp = 2;
        const unsigned int degreeTemp = 2;

        MappingQ<dimTemp> mapping(reinit_data.degree);

        QGauss<1> quad(reinit_data.degree+1);
        
        typedef VectorizedArray<double>     VectorizedArrayType ;
        typename MatrixFree<dimTemp, double, VectorizedArrayType>::AdditionalData
          additional_data;
        additional_data.mapping_update_flags = update_values | update_gradients;

        MatrixFree<dimTemp, double, VectorizedArrayType> matrix_free;

        matrix_free.reinit(mapping, *dof_handler, *constraints, quad, additional_data);
       
        solution_normal_vector.update_ghost_values();

        LevelSetMatrixFree::ReinitializationOperator<dimTemp,degreeTemp,double> rei(matrix_free,
                                                                          reinit_data.min_cell_size/(std::sqrt(2.)*2.),
                                                                          reinit_data.d_tau,
                                                                           solution_normal_vector
                                                                           );
        LinearAlgebra::distributed::Vector<double> src, dst;
        LinearAlgebra::distributed::Vector<double> solution; 

        matrix_free.initialize_dof_vector(src);
        matrix_free.initialize_dof_vector(dst);


        matrix_free.initialize_dof_vector(solution);
        

        // very dirty programming to initialize phi  @@@@@@@@@@@@@@@@@
        auto local_owners = src.locally_owned_elements();

        int idx =0;
        for (auto i = local_owners.begin(); i < local_owners.end(); ++i) 
        {
            const auto g_idx = local_owners.nth_index_in_set(idx);
            solution[g_idx] = ls[g_idx];
            idx += 1;
        }

        std::cout << "norm solution " << solution.l2_norm() << std::endl;

        ReductionControl     reduction_control;
        SolverCG<LinearAlgebra::distributed::Vector<double>> solver(reduction_control);

        std::shared_ptr<TimeIterator> time_iterator = std::make_shared<TimeIterator>();
        initialize_time_iterator(time_iterator); 
        while ( !time_iterator->is_finished() )
        {
            const double d_tau = time_iterator->get_next_time_increment();  
            std::cout << "d_tau " << d_tau << std::endl;
            rei.set_time_increment(d_tau);
            
            // create right hand side
            dst = 0.0;
            rei.create_rhs(dst, solution);

            std::cout << " RHS.norm" << dst.l2_norm() << std::endl;
            src = 0.0;
            solver.solve(rei,
                         src,
                         dst,
                         PreconditionIdentity());
            
            solution += src;
            solution.update_ghost_values();

            pcout << "   with " << reduction_control.last_step() << " CG iterations.";
            pcout << "\t |ΔΨ|∞ = " << std::setprecision(10) << src.linfty_norm() << "\t |ΔΨ|²/dT = " << std::setprecision(10) << src.l2_norm()/d_tau << std::endl;
        }
        pcout << "----------- END MATRIX FREE " << std::endl;
        std::cout << "( reinitialized ) norm solution " << solution.l2_norm() << std::endl;
    }

    template <int dim>
    void 
    Reinitialization<dim>::solve_olsson_model( VectorType & solution_out )
    {

        pcout << "       >>>>>>>>>>>>>>>>>>> REINITIALIZATION START NEW" << std::endl;

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
        std::cout << "norm solution " << solution_in.l2_norm() << std::endl;
        
        std::shared_ptr<TimeIterator> time_iterator = std::make_shared<TimeIterator>();

        initialize_time_iterator(time_iterator); 
        
        while ( !time_iterator->is_finished() )
        {
            const double d_tau = time_iterator->get_next_time_increment();
            std::cout << "d_tau " << d_tau << std::endl;
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
            
            std::cout << " RHS.norm" << system_rhs.l2_norm() << std::endl;
            
            // @ here is space for iimprovementn
            VectorType    re_solution_u_temp( locally_owned_dofs,
                                               mpi_commun );
            VectorType    re_delta_solution_u(locally_owned_dofs,
                                               mpi_commun );
            re_solution_u_temp = solution_out;
            

            /* 
             *  iterative solver
             */

            SolverControl solver_control( dof_handler->n_dofs() * 2, 1e-6 * system_rhs.l2_norm() );
            
            LA::SolverCG solver( solver_control, mpi_commun );

            LA::MPI::PreconditionAMG preconditioner;
            LA::MPI::PreconditionAMG::AdditionalData data;
            preconditioner.initialize(system_matrix, data);
            
            
            solver.solve( system_matrix, 
                          re_delta_solution_u, 
                          system_rhs, 
                          preconditioner );
             pcout << "   with " << solver_control.last_step() << " CG iterations.";


            constraints->distribute( re_delta_solution_u );

            re_solution_u_temp += re_delta_solution_u;
            
            solution_out = re_solution_u_temp;
            solution_out.update_ghost_values();

            //time_iterator->print_me( pcout.get_stream() );
            if(reinit_data.do_print_l2norm)
                pcout << "\t |ΔΨ|∞ = " << std::setprecision(10) << re_delta_solution_u.linfty_norm() << "\t |ΔΨ|²/dT = " << std::setprecision(10) << re_delta_solution_u.l2_norm()/d_tau << std::endl;

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
    //template class Reinitialization<3>; // temporarily disabled to work on matrixfree implementation
} // namespace LevelSetParallel


