#pragma once
// dealii
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
// MeltPoolDG

namespace MeltPoolDG
{
namespace ReinitializationNew
{
    using namespace dealii;

    template <int dim, int degree, typename number=double>
    class ReinitializationOperatorBase
    {
      protected:
        using VectorType          = LinearAlgebra::distributed::Vector<number>;          
        using BlockVectorType     = LinearAlgebra::distributed::BlockVector<number>;
        using VectorizedArrayType = VectorizedArray<number>;                   
        using vector              = Tensor<1, dim, VectorizedArray<number>>;                  
        using scalar              = VectorizedArray<number>;                                  
        using SparsityPatternType = TrilinosWrappers::SparsityPattern; 
        using SparseMatrixType    = TrilinosWrappers::SparseMatrix;

      public:
        
        ReinitializationOperatorBase( const double d_tau_in,
                                      const NormalVector<dim,degree>& n_in,
                                      const FE_Q<dim>&  fe_in,
                                      const MappingQGeneric<dim>& mapping_in,
                                      const QGauss<dim>& q_gauss_in,
                                      SmartPointer<const DoFHandler<dim>> dof_handler_in,
                                      SmartPointer<const AffineConstraints<number>> constraints_in
                                    )
        : d_tau               ( d_tau_in   )
        , normal_vector_field ( n_in       )
        , fe                  ( fe_in      )
        , mapping             ( mapping_in )
        , q_gauss             ( q_gauss_in ) 
        , dof_handler         ( dof_handler_in) 
        , constraints         ( constraints_in) 
        {
        }

        virtual ~ReinitializationOperatorBase() = default;

        virtual
        void
        initialize_dof_vector(VectorType &dst) const = 0;
        
        virtual void create_rhs(VectorType & dst,
                               const VectorType & src) const
        {
          AssertThrow(false, ExcMessage("create_rhs for the requested operator not implemented"));
        }

        virtual void vmult(VectorType & dst,
                           const VectorType & src) const
        {
          AssertThrow(false, ExcMessage("vmult for the requested operator not implemented"));
        }
  
        virtual void print_me()
        {
          std::cout << "hello from base class" << std::endl;
        }

        void 
        set_time_increment(const double dt)
        {
          d_tau = dt;
        }

        double                                          d_tau; 
        const NormalVector<dim,degree>&                 normal_vector_field;
        const FE_Q<dim>&                                fe;
        const MappingQGeneric<dim>&                     mapping;
        const QGauss<dim>&                              q_gauss;
        SmartPointer<const DoFHandler<dim>>             dof_handler;
        SmartPointer<const AffineConstraints<number>>   constraints;
    };
} // namespace ReinitializationNew
} // namespace MeltPoolDG
