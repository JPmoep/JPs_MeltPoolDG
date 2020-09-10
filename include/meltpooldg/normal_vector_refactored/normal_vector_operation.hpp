/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, August 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
 //for using smart pointers
#include <deal.II/base/smartpointer.h>

// MeltPoolDG
#include <meltpooldg/utilities/utilityfunctions.hpp>
#include <meltpooldg/utilities/linearsolve.hpp>
#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/normal_vector_refactored/normal_vector_operator.hpp>

namespace MeltPoolDG
{
namespace NormalVectorNew
{
  using namespace dealii; 

  /*
   *    Data for computing the normal vector of a given scalar field 
   *    considering diffusive damping;
   *
   *    !!!! 
   *          the normal vector field is not normalized to length one, 
   *          it actually represents the gradient of the level set 
   *          function 
   *    !!!! 
   */
  
  struct NormalVectorData 
  {
    // parameter for diffusive term in computation of normals
    double damping_parameter = 1e-6;
    
    // this parameter controls whether the l2 norm is printed (mainly for testing purposes)
    bool do_print_l2norm = true;
    
    // this parameter controls whether the matrixfree operator is called
    bool do_matrix_free = false;
    
    // current verbosity level --> see possible options in UtilityFunctions
    TypeDefs::VerbosityType verbosity_level = TypeDefs::VerbosityType::silent;

  };
  
  /*
   *     Model for computing the normal vector to a scalar function as a smooth function
   *     @ todo: add equation 
   */
  
  template <int dim, int degree>
  class NormalVectorOperation
  {
  private:
    using VectorType          = LinearAlgebra::distributed::Vector<double>;         
    using BlockVectorType     = LinearAlgebra::distributed::BlockVector<double>;    
    using SparseMatrixType    = TrilinosWrappers::SparseMatrix;                     
    using DoFHandlerType      = DoFHandler<dim>;                                    
    using SparsityPatternType = TrilinosWrappers::SparsityPattern;
    using ConstraintsType     = AffineConstraints<double>;

  public:
    NormalVectorData normal_vector_data;
    /*
     *    This is the primary solution variable of this module, which will be also publically 
     *    accessible for output_results.
     */
    BlockVectorType solution_normal_vector;
    
    NormalVectorOperation( const DoFHandlerType&       dof_handler_in,
                           const MappingQGeneric<dim>& mapping_in,
                           const FE_Q<dim>&            fe_in,
                           const QGauss<dim>&          q_gauss_in, 
                           const ConstraintsType&      constraints_in,
                           const IndexSet&             locally_owned_dofs_in,
                           const IndexSet&             locally_relevant_dofs_in,
                           const double                min_cell_size_in)
    :
    fe                    ( fe_in),
    mapping               ( mapping_in),
    q_gauss               ( q_gauss_in),
    dof_handler           ( &dof_handler_in ),
    constraints           ( &constraints_in ),
    locally_owned_dofs    ( locally_owned_dofs_in ),
    locally_relevant_dofs ( locally_relevant_dofs_in ),
    min_cell_size         ( min_cell_size_in ),
    mpi_communicator      ( UtilityFunctions::get_mpi_comm(*dof_handler) ),
    pcout                 ( std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
    }

    void
    initialize( const Parameters<double>& data_in )
    {
      /*
       *  initialize normal_vector data
       */
      normal_vector_data.damping_parameter = min_cell_size * data_in.normal_vec_damping_scale_factor;
      normal_vector_data.verbosity_level   = TypeDefs::VerbosityType::major;
      normal_vector_data.do_print_l2norm   = true;

      /*
       *  initialize operator
       */
      create_operator();
    }

    void
    solve( const VectorType& solution_levelset_in)
    {
      BlockVectorType rhs;
      normal_vector_operator->print_me();
      normal_vector_operator->initialize_dof_vector(rhs);
      normal_vector_operator->initialize_dof_vector(solution_normal_vector);
      
      int iter = 0;
      
      if (normal_vector_data.do_matrix_free)
      {
        normal_vector_operator->create_rhs( rhs, solution_levelset_in );
        iter = LinearSolve< BlockVectorType,
                            SolverCG<BlockVectorType>,
                            OperatorBase<double, BlockVectorType, VectorType>>
                            ::solve( *normal_vector_operator,
                                     solution_normal_vector,
                                     rhs );
        solution_normal_vector.update_ghost_values();
      }
      else
      {
        
        normal_vector_operator->assemble_matrixbased( solution_levelset_in, 
                                                      system_matrix, 
                                                      rhs );

        for (unsigned int d=0; d<dim; ++d)
        {
          iter = LinearSolve<VectorType,
                             SolverCG<VectorType>,
                             SparseMatrixType>::solve( system_matrix,
                                                       solution_normal_vector.block(d),
                                                       rhs.block(d) );

          constraints->distribute(solution_normal_vector.block(d));
          solution_normal_vector.block(d).update_ghost_values();
        }
      }

      if (normal_vector_data.do_print_l2norm)
      {
        pcout <<  "| normal vector:         i=" << iter << " \t"; 
        for(unsigned int d=0; d<dim; ++d)
          pcout << "|n_" << d << "| = " << std::setprecision(11) << std::setw(15) << std::left << solution_normal_vector.block(d).l2_norm();
        pcout << std::endl;
      }
    }

    void 
    print_me(); 
  
  private:
    void create_operator()
    {
      if (!normal_vector_data.do_matrix_free)
      {
        dsp.reinit( locally_owned_dofs,
                    locally_owned_dofs,
                    locally_relevant_dofs,
                    mpi_communicator);

        DoFTools::make_sparsity_pattern(*dof_handler, 
                                        dsp,
                                        *constraints,
                                        true,
                                        Utilities::MPI::this_mpi_process(mpi_communicator)
                                        );
        dsp.compress();
        
        system_matrix.reinit( dsp );  
      }

      normal_vector_operator = std::make_unique<NormalVectorOperator<dim, degree>>(
                                                          fe,
                                                          mapping,
                                                          q_gauss,
                                                          dof_handler,
                                                          constraints,
                                                          normal_vector_data.damping_parameter );
    }

    const FE_Q<dim>&                           fe;
    const MappingQGeneric<dim>&                mapping;
    const QGauss<dim>&                         q_gauss;
    SmartPointer<const DoFHandlerType>         dof_handler;
    SmartPointer<const ConstraintsType>        constraints;
    const IndexSet&                            locally_owned_dofs;
    const IndexSet&                            locally_relevant_dofs;
    double                                     min_cell_size;           // @todo: check CFL condition
    const MPI_Comm                             mpi_communicator;
    ConditionalOStream                         pcout;                   // @todo: reference
    /* 
     *  This pointer will point to your user-defined normal vector operator.
     */
    std::unique_ptr<OperatorBase<double, BlockVectorType, VectorType>>     
                                               normal_vector_operator;    
    
    SparseMatrixType                           system_matrix;
    SparsityPatternType                        dsp;
    //BlockVectorType                          system_rhs;
  };
} // namespace NormalVectorNew
} // namespace MeltPoolDG
