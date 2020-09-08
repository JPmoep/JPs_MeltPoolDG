#pragma once
// dealii
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/point.h>

namespace MeltPoolDG
{
using namespace dealii;

namespace TypeDefs
{
  enum class VerbosityType {silent, major, detailed};
} // namespace TypeDefs

namespace UtilityFunctions
{


namespace CharacteristicFunctions
{
  inline double tanh_characteristic_function(const double& distance, const double& eps)
  {
    return std::tanh( distance / ( 2. * eps ) );  
  }
  
  inline double heaviside(const double& distance, const double& eps)
  {
    if ( distance > eps )
        return 1;
    else if ( distance < -eps )
        return 0;
    else
        return ( distance + eps ) / ( 2.*eps )+ 1. / ( 2. * numbers::PI ) * std::sin(numbers::PI * distance / eps);
  }
  
  inline int sgn(const double& x) { return (x<0) ? -1 : 1; }

  inline double normalize(const double& x, const double& x_min, const double& x_max)
  {
    return ( x - x_min ) / ( x_max - x_min );
  }

} // namespace CharacteristicFunctions

namespace DistanceFunctions
{
  template <int dim>
  inline double spherical_manifold(const Point<dim>& p, const Point<dim>& center, const double radius)
  {
    if (dim==3)
      return -std::sqrt( std::pow(p[0]-center[0],2) + std::pow(p[1]-center[1],2) + std::pow(p[2]-center[2],2)) + radius;
    else if (dim==2)
      return -std::sqrt(std::pow(p[0]-center[0],2) + std::pow(p[1]-center[1],2)) + radius;
    else
      AssertThrow(false, ExcMessage("Spherical manifold: dim must be 2 or 3."))
  }
  
  //@todo: this function should be added to accept lines and planar surfaces
  template <int dim>
  inline double signed_distance_planar_manifold(const Point<dim>& p, const Point<dim>& start, const Point<dim>& end);
} // namespace DistanceFunctions
} // namespace UtilityFunctions 
} // namespace MeltPoolDG
    

