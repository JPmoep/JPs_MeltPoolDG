#pragma once
// dealii
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
// MeltPoolDG

namespace MeltPoolDG
{
    using namespace dealii;

    template <typename number           = double,
              typename DoFVectorType    = LinearAlgebra::distributed::Vector<number>,
              typename SrcRhsVectorType = DoFVectorType>
    class OperatorBase
    {
      private:
        using SparseMatrixType = TrilinosWrappers::SparseMatrix;
        using VectorType       = LinearAlgebra::distributed::Vector<number>;
        using BlockVectorType  = LinearAlgebra::distributed::BlockVector<number>;
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

        double d_tau = 0.0; 
    };
} // namespace MeltPoolDG
