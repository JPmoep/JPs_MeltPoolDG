
namespace LevelSetParallel
{
    using namespace dealii;

    class OutputGenerator
    {
    private:
        typedef LA::MPI::Vector                           VectorType;
        typedef LA::MPI::BlockVector                      BlockVectorType;
        typedef LA::MPI::SparseMatrix                     SparseMatrixType;

        typedef DoFHandler<dim>                           DoFHandlerType;
        
        typedef DynamicSparsityPattern                    SparsityPatternType;
        
        typedef AffineConstraints<double>                 ConstraintsType;
        
    public: 
        OutputGenerator( );
    };


}
