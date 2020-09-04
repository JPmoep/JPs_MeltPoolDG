#pragma once
// dealii
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/point.h>
#include <deal.II/numerics/data_postprocessor.h>
// c++
#include <fstream>
#include <iostream>


/*
 *  @todo: cleanup!!!
 *
 */


namespace MeltPoolDG
{
namespace UtilityFunctions
{
    using namespace dealii;
    
    typedef enum {silent=0, major=1, detailed=2} VerbosityType;
    
    namespace BoundaryConditions
    {
      enum Types { inflow=1,
                     outflow=2,
                     dirichlet=3,
                     };
      static const Types All[] = { inflow, outflow, dirichlet };
    }   

    void printSparseMatrix(const SparseMatrix<double>& mass_matrix);
    
    void printFullMatrix(const FullMatrix<double>& mass_matrix);
    
    double tanHyperbolicusCharacteristicFunction(const double& d, const double& eps);
    
    double heavisideFunction(const double& x, const double& eps);
    
    double signFunction(const double& x);
  
    double normalizeFunction(const double& x, const double& x_min, const double& x_max);

    // @ todo: sphere and circle could be merged into a general sphere using a template function 
    double signedDistanceSphere(const Point<3>& P, const Point<3>& Center, const double radius);
    
    double signedDistanceCircle(const Point<2>& P, const Point<2>& Center, const double radius);

    double signedDistanceVerticalLine(const Point<2>& P, const double xInterface);
    
    double evaluateCFLCondition();

    void printLine(const int verbosityLevel=0, std::ostream& str=std::cout, const MPI_Comm& mpi_comm=MPI_COMM_WORLD);

    template <int dim>
    class GradientPostprocessor : public DataPostprocessorVector<dim>
    {
    public:
      GradientPostprocessor ()
        :
        // call the constructor of the base class. call the variable to
        // be output "grad_u" and make sure that DataOut provides us
        // with the gradients:
        DataPostprocessorVector<dim> ("grad_u",
                                      update_gradients)
      {}
      virtual
      void
      evaluate_scalar_field(
        const DataPostprocessorInputs::Scalar<dim> &input_data,
        std::vector<Vector<double> >               &computed_quantities) const
      {
        // ensure that there really are as many output slots
        // as there are points at which DataOut provides the
        // gradients:
        AssertDimension (input_data.solution_gradients.size(),
                         computed_quantities.size());
        // then loop over all of these inputs:
        for (unsigned int p=0; p<input_data.solution_gradients.size(); ++p)
          {
            // ensure that each output slot has exactly 'dim'
            // components (as should be expected, given that we
            // want to create vector-valued outputs), and copy the
            // gradients of the solution at the evaluation points
            // into the output slots:
            AssertDimension (computed_quantities[p].size(), dim);
            for (unsigned int d=0; d<dim; ++d)
              computed_quantities[p][d]
                = input_data.solution_gradients[p][d];
          }
      }
    };

    /*
    template <int dim>
    class VectorFieldPostprocessor : public DataPostprocessorVector<dim>
    {
    public:
      VectorFieldPostprocessor ()
        :
        // call the constructor of the base class. call the variable to
        // be output "grad_u" and make sure that DataOut provides us
        // with the gradients:
        DataPostprocessorVector<dim> ("normal",
                                      update_values)
      {}
      virtual
      void
      evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim> &input_data,
        std::vector<Vector<double> >               &computed_quantities) const
      {
        // ensure that there really are as many output slots
        // as there are points at which DataOut provides the
        // gradients:
        //AssertDimension (input_data.solution_values.block(0).size(),
                         //computed_quantities.size());

        std::cout << "input_data size" << input_data.solution_values.size();
        //// then loop over all of these inputs:
        for (unsigned int p=0; p<input_data.solution_values.size(); ++p)
          {
            for (unsigned int d=0; d<dim; ++d)
            {
              std::cout << "input: " << input_data.solution_values[p][d] << std::endl;
              computed_quantities[p][d]
                = 0; // input_data.solution_values[p][d];
            }  
        }
      }
    };
    */

} // namespace UtilityFunctions
} // namespace MeltPoolDG
