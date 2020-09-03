/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, August 2020
 *
 * ---------------------------------------------------------------------*/
// for FEValues<dim>
//#include <deal.II/fe/fe.h>
// to access minimal_cell_diamater
#include <deal.II/grid/grid_tools.h>
// to use preconditioner 
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/solver_cg.h> // only for symmetric matrices
#include <deal.II/lac/trilinos_sparsity_pattern.h>
// MeltPoolDG
#include "normalvector.hpp"
#include "curvature.hpp"
#include "linearsolve.hpp"

namespace MeltPoolDG
{
    using namespace dealii; 

    template <int dim, int degree>
    Curvature<dim,degree>::Curvature(const MPI_Comm & mpi_commun_in)
    : mpi_commun( mpi_commun_in )
    , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_commun_in) == 0)
    , normal_vector_field( mpi_commun_in )
    {
    }

    template <int dim, int degree>
    void
    Curvature<dim,degree>::initialize( const CurvatureData &       data_in,
                                       const SparsityPatternType&  dsp_in,
                                       const DoFHandler<dim>&      dof_handler_in,
                                       const ConstraintsType&      constraints_in,
                                       const IndexSet&             locally_owned_dofs_in,
                                       const IndexSet&             locally_relevant_dofs_in
                                    )
    {
        curvature_data        = data_in;
        dof_handler           = &dof_handler_in;
        constraints           = &constraints_in;
        locally_owned_dofs    = locally_owned_dofs_in;
        locally_relevant_dofs = locally_relevant_dofs_in;
        
        system_matrix.reinit( dsp_in );
        
        system_rhs.reinit( locally_owned_dofs, 
                           locally_relevant_dofs,
                           mpi_commun ); 
        
        /*
         * here the current verbosity level is set
         */
        const bool verbosity_active = ((Utilities::MPI::this_mpi_process(mpi_commun) == 0) && (curvature_data.verbosity_level!=utilityFunctions::VerbosityType::silent));
        this->pcout.set_condition(verbosity_active);

        /*
         * initialize the normal_vector_field computation
         * @ todo: how should data be transferred from the base class ?
         */

        NormalVectorData normal_vector_data;
        normal_vector_data.damping_parameter = curvature_data.min_cell_size * 0.5;
        normal_vector_data.degree            = curvature_data.degree;
        normal_vector_data.verbosity_level   = curvature_data.verbosity_level;
        //normal_vector_data.min_cell_size     = curvature_data.min_cell_size;
        //normal_vector_data.do_print_l2norm   = curvature_data.do_print_l2norm;

        normal_vector_field.initialize( normal_vector_data, 
                                        dsp_in,
                                        dof_handler_in,
                                        constraints_in,
                                        locally_owned_dofs_in,
                                        locally_relevant_dofs_in,
                                        curvature_data.min_cell_size );
    }
    
    template <int dim, int degree>
    void 
    Curvature<dim,degree>::solve( const VectorType & solution_in,
                                 VectorType & curvature_out )
    {
        //TimerOutput::Scope timer (computing_timer, "Curvature computation.");
        
        normal_vector_field.compute_normal_vector_field(solution_in);

        system_matrix = 0.0; 
        system_rhs = 0.0;

        FE_Q<dim> fe( curvature_data.degree );

        auto qGauss = QGauss<dim>( curvature_data.degree+1 );

        FEValues<dim> fe_values( fe,
                                 qGauss,
                                 update_values | update_gradients | update_quadrature_points | update_JxW_values );

        const unsigned int n_q_points    = qGauss.size();
        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        
        std::vector<types::global_dof_index> local_dof_indices(     dofs_per_cell );
        FullMatrix<double>                   curvature_cell_matrix( dofs_per_cell, dofs_per_cell );
        Vector<double>                       curvature_cell_rhs(    dofs_per_cell );

        const double curvature_damping = curvature_data.damping_parameter; //@todo: modifiy damping parameter

        for (const auto &cell : dof_handler->active_cell_iterators())
        if (cell->is_locally_owned())
        {
            fe_values.reinit( cell );
            
            curvature_cell_matrix = 0.0;
            curvature_cell_rhs    = 0.0;
            
            std::vector<Tensor<1,dim>>           normal_at_q( n_q_points, Tensor<1,dim>() );

            normal_vector_field.get_unit_normals_at_quadrature( fe_values,
                                                                normal_at_q );
            
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
                    curvature_cell_rhs(i) += grad_phi_i
                                             * 
                                             normal_at_q[ q_index ] 
                                             * 
                                             fe_values.JxW( q_index );
                }
            }
            
            cell->get_dof_indices(local_dof_indices);
            constraints->distribute_local_to_global( curvature_cell_matrix,
                                                     curvature_cell_rhs,
                                                     local_dof_indices,
                                                     system_matrix,
                                                     system_rhs);
        } // end loop over cells
        system_matrix.compress( VectorOperation::add );
        system_rhs.compress(    VectorOperation::add );
        
        curvature_out.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_commun);
        LinearSolve<VectorType,SolverCG<VectorType>>::solve( system_matrix,
                                                             curvature_out,
                                                             system_rhs );
        constraints->distribute(curvature_out);
        curvature_out.update_ghost_values();
    }
    
    template <int dim, int degree>
    void 
    Curvature<dim,degree>::solve( const VectorType & solution_in )
    {
        curvature_field.reinit( locally_owned_dofs, 
                                locally_relevant_dofs,
                                mpi_commun ); 

        solve( solution_in, curvature_field);
    }
    
    template <int dim, int degree>
    typename Curvature<dim,degree>::VectorType
    Curvature<dim,degree>::get_curvature_values( ) const
    {
        return curvature_field;
    }

    template <int dim, int degree>
    void
    Curvature<dim,degree>::print_me( )
    {
        pcout << "hello from curvature computation" << std::endl;   
        pcout << "damping: "              << curvature_data.damping_parameter << std::endl;
        pcout << "degree: "               << curvature_data.degree            << std::endl;
    }

    // instantiation
    template class Curvature<2,1>;
    template class Curvature<2,2>;
    //template class Curvature<3>;

} // namespace MeltPoolDG


