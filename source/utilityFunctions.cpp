#include "utilityFunctions.hpp"
#include <math.h>

using namespace dealii;

namespace utilityFunctions
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
  
  double tanHyperbolicusCharacteristicFunction(const double& d, const double& eps)
  {
        //return std::tanh( d / ( eps ) ); 
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

} // end of utilityFunctions
