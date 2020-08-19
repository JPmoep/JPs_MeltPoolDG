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
#include <normalvector.hpp>
#include <deal.II/fe/mapping.h>


namespace LevelSetParallel
{
    using namespace dealii; 

    template <int dim>
    NormalVector<dim>::NormalVector(const MPI_Comm & mpi_commun_in)
    : mpi_commun( mpi_commun_in )
    , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_commun) == 0)
    {
    }

    template <int dim>
    void
    NormalVector<dim>::initialize( const NormalVectorData &      data_in,
                                   const SparsityPatternType&    dsp_in,
                                   DoFHandler<dim> const &       dof_handler_in,
                                   const ConstraintsType&        constraints_in,
                                   const IndexSet&               locally_owned_dofs_in
                                    )
    {
        normal_vector_data = data_in;
        dof_handler        = &dof_handler_in;
        constraints        = &constraints_in;
        locally_owned_dofs = locally_owned_dofs_in;
        
        system_matrix.reinit( locally_owned_dofs,
                              locally_owned_dofs,
                              dsp_in,
                              mpi_commun );
        
        // @ is there a better way to reinitialize a block vector??
        system_rhs.reinit( dim );
        for (unsigned int d=0; d<dim; ++d)
            system_rhs.block(d).reinit( locally_owned_dofs, 
                                        mpi_commun ); 
        
        const bool verbosity_active = ((Utilities::MPI::this_mpi_process(mpi_commun) == 0) && (normal_vector_data.verbosity_level!=utilityFunctions::VerbosityType::silent));
        this->pcout.set_condition(verbosity_active);
    }

    template <int dim>
    void 
    NormalVector<dim>::solve( const VectorType & solution_in,
                                    BlockVectorType & normal_vector_out )
    {
        //TimerOutput::Scope t(computing_timer, "compute damped normals");  
        system_matrix = 0.0;
        for (unsigned int d=0; d<dim; d++)
            system_rhs.block(d) = 0.0;

        auto qGauss = QGauss<dim>(normal_vector_data.degree+1);
        
        FE_Q<dim> fe(normal_vector_data.degree);
        
        FEValues<dim> fe_values( fe,
                                 qGauss,
                                 update_values | update_gradients | update_quadrature_points | update_JxW_values );
        const unsigned int n_q_points    = qGauss.size();

        const unsigned int          dofs_per_cell = fe.dofs_per_cell;
        FullMatrix<double>          normal_cell_matrix( dofs_per_cell, dofs_per_cell );
        std::vector<Vector<double>> normal_cell_rhs(    dim, Vector<double>(dofs_per_cell) );
        
        std::vector<types::global_dof_index> local_dof_indices( dofs_per_cell );
        std::vector<Tensor<1,dim>>           normal_at_q(  n_q_points, Tensor<1,dim>() );

        double damping = normal_vector_data.damping_parameter; //GridTools::minimal_cell_diameter(triangulation) * 0.5; //@todo: modifiy damping parameter
        
        for (const auto &cell : dof_handler->active_cell_iterators())
        if (cell->is_locally_owned())
        {
            fe_values.reinit(cell);
            cell->get_dof_indices( local_dof_indices );
            
            normal_cell_matrix = 0.0;
            for(auto& normal_cell : normal_cell_rhs)
                normal_cell =    0.0;

            fe_values.get_function_gradients( solution_in, normal_at_q ); // compute normals from level set solution at tau=0
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
                constraints->distribute_local_to_global(normal_cell_matrix,
                                                          normal_cell_rhs[d],
                                                          local_dof_indices,
                                                          system_matrix,
                                                          system_rhs.block(d));
             
          }
          system_matrix.compress(VectorOperation::add);
          for (unsigned int d=0; d<dim; ++d)
            system_rhs.block(d).compress(VectorOperation::add);
        
        // @todo
        //for (unsigned int d=0; d<dim; ++d)
        //{
            //solve_cg(system_rhs.block( d ), system_matrix, normal_vector_out.block( d ), "damped normals");
            //constraints->distribute(normal_vector_out.block( d ));
       //}

    }


    template <int dim>
    void
    NormalVector<dim>::print_me( )
    {
        pcout << "hello from normal vector computation" << std::endl;   
        pcout << "damping: "              << normal_vector_data.damping_parameter << std::endl;
        pcout << "degree: "               << normal_vector_data.degree            << std::endl;
    }

    // instantiation
    template class NormalVector<2>;
    template class NormalVector<3>;
} // namespace LevelSetParallel


