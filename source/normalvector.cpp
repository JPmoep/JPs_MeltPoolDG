/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, August 2020
 *
 * ---------------------------------------------------------------------*/
// for FEValues<dim>
//#include <deal.II/fe/fe.h>
// to access minimal_cell_diameter
#include <deal.II/grid/grid_tools.h>
// to use preconditioner 
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/solver_cg.h> // only for symmetric matrices
// for matrix free solution
#include <deal.II/fe/mapping_q.h>
#include <deal.II/matrix_free/matrix_free.h>
// multiphaseflow
#include <normalvectoroperator.hpp>
#include <normalvector.hpp>
#include <linearsolve.hpp>

namespace MeltPoolDG
{
    using namespace dealii; 

    template <int dim, int degree>
    NormalVector<dim,degree>::NormalVector(const MPI_Comm & mpi_commun_in)
    : mpi_commun( mpi_commun_in )
    , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_commun) == 0)
    {
    }

    // @ todo: better call initialize_as_submodule??
    template <int dim, int degree>
    void
    NormalVector<dim,degree>::initialize( const NormalVectorData &   data_in,
                                          const SparsityPatternType& dsp_in,
                                          const DoFHandlerType&      dof_handler_in,
                                          const ConstraintsType&     constraints_in,
                                          const IndexSet&            locally_owned_dofs_in,
                                          const IndexSet&            locally_relevant_dofs_in,
                                          const double               min_cell_size_in
                                 )
    {
        // replace by extract_local_ ....
        normal_vector_data    = data_in;
        dof_handler           = &dof_handler_in;
        constraints           = &constraints_in;
        locally_owned_dofs    = locally_owned_dofs_in;
        locally_relevant_dofs = locally_relevant_dofs_in;
        min_cell_size         = min_cell_size_in;
        
        system_matrix.reinit( dsp_in );
        
        // @ is there a better way to reinitialize a block vector??
        system_rhs.reinit( dim );
        for (unsigned int d=0; d<dim; ++d)
            system_rhs.block(d).reinit( locally_owned_dofs, 
                                        locally_relevant_dofs,
                                        mpi_commun ); 
        /*
         * the current verbosity level is set
         */
        const bool verbosity_active = ((Utilities::MPI::this_mpi_process(mpi_commun) == 0) && (normal_vector_data.verbosity_level!=utilityFunctions::VerbosityType::silent));
        this->pcout.set_condition(verbosity_active);
    }

    template <int dim, int degree>
    void 
    NormalVector<dim,degree>::extract_local_parameters_from_global_parameters( const LevelSetParameters& param_in)
    {
         //@ introduce new C++20 features --> shift to normalvector class
      NormalVectorData normal_vector_data;
      normal_vector_data.damping_parameter = this->min_cell_size * 0.5;
      normal_vector_data.verbosity_level   = utilityFunctions::VerbosityType::major; // @ introduce verbosity levels in global parameters
      normal_vector_data.do_print_l2norm   = false; //@ param_in.do_print_l2norm; 
    }


    template <int dim, int degree>
    void 
    NormalVector<dim,degree>::compute_normal_vector_field_matrixfree( const VectorType & levelset_in,
                                                                      BlockVectorType & normal_vector_out)
    {
      MappingQ<dim> mapping( degree );
      QGauss<1> quad_1d(     degree + 1 );
      
      typedef VectorizedArray<double>     VectorizedArrayType ;
      typename MatrixFree<dim, double, VectorizedArrayType>::AdditionalData
        additional_data;

      additional_data.mapping_update_flags = update_values | update_gradients;

      MatrixFree<dim, double, VectorizedArrayType> matrix_free;

      matrix_free.reinit(mapping, *dof_handler, *constraints, quad_1d, additional_data);
      
      typedef LevelSetMatrixFree::NormalVectorOperator<dim, degree> OperatorType;
      OperatorType normal_operator( matrix_free,
                                    normal_vector_data.damping_parameter );

      /* copy initial level set field
       *
       *@ --> is there a better solution? if I do not copy the vector the condition
       * vec.partitioners_are_compatible(*dof_info.vector_partitioner) is violated
      */
      VectorType level; 
      matrix_free.initialize_dof_vector(level);
      level.copy_locally_owned_data_from(levelset_in);
      level.update_ghost_values();

      BlockVectorType rhs;
      normal_operator.initialize_dof_vector(rhs);
      normal_operator.initialize_dof_vector(normal_vector_out);
      
      // compute right-hand side
      normal_operator.create_rhs(rhs, level);
      
      LinearSolve<BlockVectorType,SolverCG<BlockVectorType>, OperatorType>::solve( normal_operator,
                                                                                   normal_vector_out,
                                                                                   rhs );
      normal_vector_out.update_ghost_values();

      if (normal_vector_data.do_print_l2norm)
        for(unsigned int d=0; d<dim; ++d)
          pcout << std::setprecision(10) << "   normal vector: ||n_" << d << "|| = " << normal_vector_out.block(d).l2_norm() << std::endl;
    }

    template <int dim, int degree>
    void 
    NormalVector<dim,degree>::compute_normal_vector_field_matrixfree( const VectorType & solution_in )
    {
        compute_normal_vector_field_matrixfree( solution_in, normal_vector_field );
    }

    template <int dim, int degree>
    void 
    NormalVector<dim,degree>::compute_normal_vector_field( const VectorType & solution_in,
                                                    BlockVectorType & normal_vector_out ) 
    {
      solution_in.update_ghost_values();
      //TimerOutput::Scope t(computing_timer, "compute damped normals");  
      system_matrix = 0.0;
      system_rhs=0.0;
      normal_vector_out = 0.0;

      for (unsigned int d=0; d<dim; d++)
      {
          system_rhs.block(d) = 0.0;
          normal_vector_out.block(d) = 0.0;
      } 

      auto qGauss = QGauss<dim>(normal_vector_data.degree+1);
      
      FE_Q<dim> fe(normal_vector_data.degree);
      
      FEValues<dim> fe_values( fe,
                               qGauss,
                               update_values | update_gradients | update_quadrature_points | update_JxW_values );

      const unsigned int                    dofs_per_cell = fe.dofs_per_cell;
      FullMatrix<double>                    normal_cell_matrix( dofs_per_cell, dofs_per_cell );
      std::vector<Vector<double>>           normal_cell_rhs(    dim, Vector<double>(dofs_per_cell) );
      std::vector<types::global_dof_index>  local_dof_indices( dofs_per_cell );
      
      const unsigned int n_q_points    = qGauss.size();
      std::vector<Tensor<1,dim>>           normal_at_q(  n_q_points, Tensor<1,dim>() );

      for (const auto &cell : dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        cell->get_dof_indices( local_dof_indices );
        
        normal_cell_matrix = 0.0;
        for(auto& normal_cell : normal_cell_rhs)
            normal_cell =    0.0;

        fe_values.get_function_gradients( solution_in, normal_at_q ); // compute normals from level set solution at tau=0
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
                      
                      // clang-format off
                      normal_cell_matrix( i, j ) += ( 
                                                      phi_i * phi_j 
                                                      + 
                                                      normal_vector_data.damping_parameter * grad_phi_i * grad_phi_j  
                                                    )
                                                    * 
                                                    fe_values.JxW( q_index ) ;
                      // clang-format on
                  }
     
                  for (unsigned int d=0; d<dim; ++d)
                  {
                      // clang-format off
                      normal_cell_rhs[d](i) +=   phi_i
                                                 * 
                                                 normal_at_q[ q_index ][ d ]  
                                                 * 
                                                 fe_values.JxW( q_index );
                        // clang-format on
                    }
                }
            }
            
            // assembly
            cell->get_dof_indices(local_dof_indices);

            constraints->distribute_local_to_global( normal_cell_matrix,
                                                     local_dof_indices,
                                                     system_matrix);
            for (unsigned int d=0; d<dim; ++d)
                constraints->distribute_local_to_global( normal_cell_rhs[d],
                                                         local_dof_indices,
                                                         system_rhs.block(d) );
             
        } // end of cell loop
        system_matrix.compress( VectorOperation::add );
        system_rhs.compress(    VectorOperation::add );
        
        for (unsigned int d=0; d<dim; ++d)
        {
          LinearSolve<VectorType,SolverCG<VectorType>>::solve( system_matrix,
                                                               normal_vector_out.block(d),
                                                               system_rhs.block(d));
        }

        normal_vector_out.update_ghost_values();
        if (normal_vector_data.do_print_l2norm)
          for(unsigned int d=0; d<dim; ++d)
            pcout << std::setprecision(10) << "   normal vector: ||n_" << d << "|| = " << normal_vector_out.block(d).l2_norm() << std::endl;
    }
    
    template <int dim, int degree>
    void 
    NormalVector<dim,degree>::compute_normal_vector_field( const VectorType & solution_in )
    {
        normal_vector_field.reinit( dim );
        for (unsigned int d=0; d<dim; ++d)
            normal_vector_field.block(d).reinit( locally_owned_dofs, 
                                                 locally_relevant_dofs,
                                                 mpi_commun ); 

        compute_normal_vector_field( solution_in, normal_vector_field );
    }

    template <int dim, int degree>
    void
    NormalVector<dim,degree>::get_unit_normals_at_quadrature( const FEValues<dim>& fe_values,
                                                       const BlockVectorType& normal_vector_field_in,
                                                       std::vector<Tensor<1,dim>>& unit_normal_at_quadrature ) const
    {
        for (unsigned int d=0; d<dim; ++d )
        {
            std::vector<double> temp ( unit_normal_at_quadrature.size() );
            fe_values.get_function_values(  normal_vector_field_in.block(d), temp); // compute normals from level set solution at tau=0
            for (const unsigned int q_index : fe_values.quadrature_point_indices())
                unit_normal_at_quadrature[ q_index ][ d ] = temp[ q_index ];
        }
        for (auto& n : unit_normal_at_quadrature)
            n /= n.norm(); //@todo: add exception
    }

    template <int dim, int degree>
    void
    NormalVector<dim,degree>::get_unit_normals_at_quadrature( const FEValues<dim>& fe_values,
                                                       std::vector<Tensor<1,dim>>& unit_normal_at_quadrature ) const
    {
        get_unit_normals_at_quadrature(fe_values, normal_vector_field, unit_normal_at_quadrature );
    }

    template <int dim, int degree>
    void
    NormalVector<dim,degree>::print_me( )
    {
        pcout << "hello from normal vector computation"                           << std::endl;   
        pcout << "damping: "              << normal_vector_data.damping_parameter << std::endl;
        pcout << "degree: "               << normal_vector_data.degree            << std::endl;
    }

    // instantiation
    template class NormalVector<2,1>;
    template class NormalVector<2,2>;

} // namespace MeltPoolDG
