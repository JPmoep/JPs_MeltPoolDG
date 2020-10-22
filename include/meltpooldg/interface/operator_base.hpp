#pragma once
// dealii
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
// MeltPoolDG
#include <meltpooldg/interface/scratch_data.hpp>

namespace MeltPoolDG
{
    using namespace dealii;

    template< typename number           = double,
              typename DoFVectorType    = LinearAlgebra::distributed::Vector<number>,
              typename SrcRhsVectorType = DoFVectorType>
    class OperatorBase
    {
      private:
        using SparseMatrixType    = TrilinosWrappers::SparseMatrix;
        using SparsityPatternType = TrilinosWrappers::SparsityPattern;
      
      public:

        virtual ~OperatorBase() = default;
        
        virtual 
        void 
        assemble_matrixbased(const SrcRhsVectorType & src,
                             SparseMatrixType & matrix,
                             DoFVectorType & rhs) const
        {
          (void)src;
          (void)matrix;
          (void)rhs;
          AssertThrow(false, ExcMessage("assemble_matrixbased for the requested operator not implemented"));
        }

        virtual void create_rhs(DoFVectorType & dst,
                               const SrcRhsVectorType & src) const
        {
          (void)dst;
          (void)src;
          AssertThrow(false, ExcMessage("create_rhs for the requested operator not implemented"));
        }

        virtual void vmult(DoFVectorType & dst,
                           const DoFVectorType & src) const
        {
          (void)dst;
          (void)src;
          AssertThrow(false, ExcMessage("vmult for the requested operator not implemented"));
        }
  
        virtual void print_me() const
        {
        }

        void 
        set_time_increment(const double dt)
        {
          d_tau = dt;
        }
        
        template<unsigned int dim,
                 unsigned int comp>
        void
        initialize_matrix_based(const ScratchData<dim>& scratch_data)
        {
          const MPI_Comm mpi_communicator = scratch_data.get_mpi_comm(comp);  
          dsp.reinit( scratch_data.get_locally_owned_dofs(),
                      scratch_data.get_locally_owned_dofs(),
                      scratch_data.get_locally_relevant_dofs(),
                      mpi_communicator);

          DoFTools::make_sparsity_pattern(scratch_data.get_dof_handler(comp), 
                                          this->dsp,
                                          scratch_data.get_constraint(comp),
                                          true,
                                          Utilities::MPI::this_mpi_process(mpi_communicator)
                                          );
          this->dsp.compress();
          
          this->system_matrix.reinit( dsp );  
        }

        template<int dim>
        void 
        create_rhs_and_apply_dirichlet_mf(DoFVectorType          &rhs,
                                          const SrcRhsVectorType &src,
                                          const ScratchData<dim> &scratch_data,
                                          const int dof_idx      =0
                                          )
        {
          DoFVectorType bc_values;
          scratch_data.initialize_bc_vector(bc_values, dof_idx);
          /*
           * perform matrix-vector multiplication (with unconstrained system and constrained set in Vector)
           */
          this->vmult(rhs, bc_values);

          /*
           * Modify right-hand side
           */
          rhs *= -1.0;
          this->create_rhs( rhs, src);

          // clear constrainted values
          scratch_data.get_constraint(dof_idx).set_zero(rhs);
        }
         
        const SparseMatrixType &
        get_system_matrix() const
        {
          return this->system_matrix;
        }

        double d_tau            = 0.0; 
        SparseMatrixType        system_matrix;
        SparsityPatternType     dsp;
    };
} // namespace MeltPoolDG
