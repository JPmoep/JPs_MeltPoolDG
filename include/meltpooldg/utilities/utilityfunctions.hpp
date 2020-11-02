#pragma once
// dealii
#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/generic_linear_algebra.h>

namespace MeltPoolDG
{
  using namespace dealii;

  namespace TypeDefs
  {
    enum class VerbosityType
    {
      silent,
      major,
      detailed
    };
  } // namespace TypeDefs

  namespace UtilityFunctions
  {
    template <typename MeshType>
    MPI_Comm
    get_mpi_comm(const MeshType &mesh)
    {
      const auto *tria_parallel =
        dynamic_cast<const parallel::TriangulationBase<MeshType::dimension> *>(
          &(mesh.get_triangulation()));

      return tria_parallel != nullptr ? tria_parallel->get_communicator() : MPI_COMM_SELF;
    }

    namespace CharacteristicFunctions
    {
      inline double
      tanh_characteristic_function(const double &distance, const double &eps)
      {
        return std::tanh(distance / (2. * eps));
      }

      inline double
      heaviside(const double &distance, const double &eps)
      {
        if (distance > eps)
          return 1;
        else if (distance < -eps)
          return 0;
        else
          return (distance + eps) / (2. * eps) +
                 1. / (2. * numbers::PI) * std::sin(numbers::PI * distance / eps);
      }

      inline int
      sgn(const double &x)
      {
        return (x < 0) ? -1 : 1;
      }

      inline double
      normalize(const double &x, const double &x_min, const double &x_max)
      {
        return (x - x_min) / (x_max - x_min);
      }

    } // namespace CharacteristicFunctions

    namespace DistanceFunctions
    {
      template <int dim>
      inline double
      spherical_manifold(const Point<dim> &p, const Point<dim> &center, const double radius)
      {
        if (dim == 3)
          return -std::sqrt(std::pow(p[0] - center[0], 2) + std::pow(p[1] - center[1], 2) +
                            std::pow(p[2] - center[2], 2)) +
                 radius;
        else if (dim == 2)
          return -std::sqrt(std::pow(p[0] - center[0], 2) + std::pow(p[1] - center[1], 2)) + radius;
        else if (dim == 1)
          return -std::sqrt(std::pow(p[0] - center[0], 2)) + radius;
        else
          AssertThrow(false, ExcMessage("Spherical manifold: dim must be 1, 2 or 3."));
      }

      template <int dim>
      inline double infinte_line(const Point<dim>& p, Point<dim>& fix_p1, Point<dim>& fix_p2)
      {
        if (dim == 3)
          return std::sqrt(std::pow((fix_p2[1]-fix_p1[1])*(fix_p1[2]-p[2]) - (fix_p2[2]-fix_p1[2])*(fix_p1[1] - p[1]), 2) + 
                          std::pow((fix_p2[2]-fix_p1[2])*(fix_p1[0]-p[0]) - (fix_p2[0]-fix_p1[0])*(fix_p1[2] - p[2]), 2) + 
                          std::pow((fix_p2[0]-fix_p1[0])*(fix_p1[1]-p[1]) - (fix_p2[1]-fix_p1[1])*(fix_p1[0] - p[0]), 2)) / 
                  std::sqrt(std::pow(fix_p2[0]-fix_p1[0],2) + std::pow(fix_p2[1]-fix_p1[1],2) +  std::pow(fix_p2[2]-fix_p1[2],2));
        else if (dim == 2)
          return std::abs((fix_p2[0] -fix_p1[0]) * (fix_p1[1] - p[1]) - (fix_p2[1]-fix_p1[1]) * (fix_p1[0] - p[0])) / 
                          std::sqrt(std::pow(fix_p2[0]-fix_p1[0],2) + std::pow(fix_p2[1]-fix_p1[1],2));  
        else if (dim == 1)
          return std::abs(fix_p1[0] - p[0]);  
        else
          AssertThrow(false, ExcMessage("distance to infinite line: dim must be 1, 2 or 3."));  

      }

      
      //@todo: this function should be added to compute distance to slotted disc, not finished
      template <int dim>
      inline double signed_distance_slotted_disc(const Point<dim>& p, const Point<dim>& center, const double radius, const double slot_w, const double slot_l)
      {
        // default distance
        double d_AB = std::numeric_limits<double>::max(); 
        double d_BC = std::numeric_limits<double>::max();
        double d_CD = std::numeric_limits<double>::max();
        double d_manifold = std::numeric_limits<double>::max();
        double d_min;
        // set corner points
        const double delta_y = radius - std::sqrt(std::pow(radius,2) - (std::pow(slot_w, 2))/4);
        Point<dim> pA     = Point<dim>(center[0] - slot_w/2, center[1] - radius + delta_y); 
        Point<dim> pB     = Point<dim>(center[0] - slot_w/2, center[1] + (slot_l - radius)); 
        Point<dim> pC     = Point<dim>(center[0] + slot_w/2, center[1] + (slot_l - radius)); 
        Point<dim> pD     = Point<dim>(center[0] + slot_w/2,  center[1] - radius + delta_y);

        if (p[1] <= pA[1]){
          if (p[0] >= pA[0] && p[0] <= pD[0]){                // region 10 and 11
            d_AB = UtilityFunctions::DistanceFunctions::spherical_manifold<dim>( p, pA, 0.0 );
            d_CD = UtilityFunctions::DistanceFunctions::spherical_manifold<dim>( p, pD, 0.0 );
            d_min = std::max(d_AB, d_CD);
          } 
          else{                                    // boundary region of 10 and 11
            d_AB = UtilityFunctions::DistanceFunctions::spherical_manifold<dim>( p, pA, 0.0 );
            d_CD = UtilityFunctions::DistanceFunctions::spherical_manifold<dim>( p, pD, 0.0 );
            d_manifold = UtilityFunctions::DistanceFunctions::spherical_manifold<dim>( p, center, radius );
            d_min = std::max(d_AB, d_CD);
            d_min = std::max(d_manifold, d_min);
          }
        } 
        else if (p[1] >= pB[1]){ 
          if (p[0] <= pB[0]){                                   // region 3
            d_BC = std::abs(UtilityFunctions::DistanceFunctions::spherical_manifold<dim>( p, pB, 0.0 ));
            d_manifold = UtilityFunctions::DistanceFunctions::spherical_manifold<dim>( p, center, radius );
            d_min = std::min(d_BC, d_manifold);
          } 
          else if (p[0] >= pC[0]){                              //region 4
            d_BC = std::abs(UtilityFunctions::DistanceFunctions::spherical_manifold<dim>( p, pC, 0.0 ));
            d_manifold = UtilityFunctions::DistanceFunctions::spherical_manifold<dim>( p, center, radius );
            d_min = std::min(d_BC, d_manifold);
          } 
          else if (p[0] > pB[0] && p[0] < pC[0]){               // region 2
            d_BC = UtilityFunctions::DistanceFunctions::infinte_line<dim>(p, pB, pC);
            d_manifold = UtilityFunctions::DistanceFunctions::spherical_manifold<dim>( p, center, radius );
            d_min = std::min(d_BC, d_manifold);
          }
        }
        else if (p[0] > center[0]-radius && p[0] < center[0]+radius)  // region 1, 5-7, 8, 9
        {
          if (p[0] > pB[0] && p[0] < pC[0])  // region 5-7
          {
            d_AB = -UtilityFunctions::DistanceFunctions::infinte_line<dim>(p, pA, pB);
            d_BC = -UtilityFunctions::DistanceFunctions::infinte_line<dim>(p, pB, pC);
            d_CD = -UtilityFunctions::DistanceFunctions::infinte_line<dim>(p, pC, pD);
            d_min = std::max(d_AB, d_BC);
            d_min = std::max(d_CD, d_min);
          }
          else
          {
            d_AB = UtilityFunctions::DistanceFunctions::infinte_line<dim>(p, pA, pB);
            d_CD = UtilityFunctions::DistanceFunctions::infinte_line<dim>(p, pC, pD);
            d_manifold = UtilityFunctions::DistanceFunctions::spherical_manifold<dim>( p, center, radius );
            d_min = std::min(d_AB, d_CD);
            d_min = std::min(d_min, d_manifold);
          }
        }
        else{                                                   // outer region
          d_min = UtilityFunctions::DistanceFunctions::spherical_manifold<dim>( p, center, radius );
        } 

        // return the sign of the smallest distance
        return UtilityFunctions::CharacteristicFunctions::sgn(d_min); 
      
      }


      //@todo: this function should be added to accept lines and planar surfaces
      template <int dim>
      inline double
      signed_distance_planar_manifold(const Point<dim> &p,
                                      const Point<dim> &start,
                                      const Point<dim> &end);
    } // namespace DistanceFunctions
  }   // namespace UtilityFunctions
} // namespace MeltPoolDG
