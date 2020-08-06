#pragma once
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/point.h>
#include <fstream>
#include <iostream>

namespace utilityFunctions
{

    using namespace dealii;
    enum BCTypes { inflow=1,
                   outflow=2,
                   dirichlet=3,
                 };
    
    void printSparseMatrix(const SparseMatrix<double>& mass_matrix);
    
    void printFullMatrix(const FullMatrix<double>& mass_matrix);
    
    double tanHyperbolicusCharacteristicFunction(const double& d, const double& eps);
    
    double heavisideFunction(const double& x, const double& eps);
    
    double signedDistanceSphere(const Point<3>& P, const Point<3>& Center, const double radius);
    
    double signedDistanceCircle(const Point<2>& P, const Point<2>& Center, const double radius);

    double signedDistanceVerticalLine(const Point<2>& P, const double xInterface);
    
    double evaluateCFLCondition();

}
