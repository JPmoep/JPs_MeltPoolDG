#include "utilityfunctions.hpp"
#include <math.h>
#include <iomanip>

using namespace dealii;

namespace MeltPoolDG
{

namespace UtilityFunctions
{

  double heavisideFunction(const double& d, const double& eps)
  {
        if ( d > eps )
            return 1;
        else if ( d < -eps )
            return 0;
        else
            return ( d + eps ) / ( 2.*eps )+ 1. / ( 2. * numbers::PI ) * std::sin(numbers::PI * d / eps);
  }
  
  double signFunction(const double& d)
  {
        return (d<0) ? -1 : 1;
  }
  
  double normalizeFunction(const double& x, const double& x_min, const double& x_max)
  {
        return ( x - x_min ) / ( x_max - x_min );
  }

  double tanHyperbolicusCharacteristicFunction(const double& d, const double& eps)
  {
        return std::tanh( d / ( 2. * eps ) ); // denominator modified to obtain only a slight shift between the intial and the first initialization step
  }

  void printSparseMatrix(const SparseMatrix<double>& mass_matrix)
  {
    std::ostringstream outputMass;
    
    for (size_t i =0; i < mass_matrix.m(); i++)
    {
        for (size_t j =0; j < mass_matrix.n(); j++)
                outputMass << mass_matrix.el(i,j) << " ";

        outputMass << std::endl;
    }
    std::cout << outputMass.str();
  }


void printFullMatrix(const FullMatrix<double>& mass_matrix)
  {
    std::ostringstream outputMass;
    
    for (size_t i =0; i < mass_matrix.m(); i++)
    {
        for (size_t j =0; j < mass_matrix.n(); j++)
                outputMass << mass_matrix(i,j) << " ";

        outputMass << std::endl;
    }
    std::cout << outputMass.str();
  }
 
double signedDistanceSphere(const Point<3>& P, const Point<3>& Center, const double radius)
 {
    return -std::sqrt( std::pow(P[0]-Center[0],2) + std::pow(P[1]-Center[1],2) + std::pow(P[2]-Center[2],2)) + radius;
 }

 double signedDistanceCircle(const Point<2>& P, const Point<2>& Center, const double radius)
 {
    return -std::sqrt(std::pow(P[0]-Center[0],2) + std::pow(P[1]-Center[1],2)) + radius;
 }
 
 double signedDistanceVerticalLine(const Point<2>& P, const double xInterface)
 {
    return P[0] - xInterface;  
 }

 void printLine(const int verbosityLevel, std::ostream& str, const MPI_Comm& mpi_comm)
 {
     if( Utilities::MPI::this_mpi_process(mpi_comm) == 0) 
          str << "+" << std::string(68, '-') << "+" << std::endl;
 }

} // namespace UtilityFunctions
} // namespace MeltPoolDG
