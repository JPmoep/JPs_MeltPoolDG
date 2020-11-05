#pragma once
// dealii
#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

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

    /**
     * This function returns heaviside values for a given VectorizedArray. The limit to
     * distinguish between 0 and 1 can be adjusted by the argument "limit". This function is
     * particularly suited in the context of MatrixFree routines.
     */
    template <typename number>
    VectorizedArray<number>
    heaviside(const VectorizedArray<number> &in, const number limit = 0.0)
    {
      return compare_and_apply_mask<SIMDComparison::greater_than>(in,
                                                                  VectorizedArray<double>(limit),
                                                                  1.0,
                                                                  0.0);
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

      

      //@todo: this function should be added to accept lines and planar surfaces
      template <int dim>
      inline double
      signed_distance_planar_manifold(const Point<dim> &p,
                                      const Point<dim> &normal,
                                      const Point<dim> &point_on_plane)
      {
        if ( (dim == 3) || (dim == 2) || (dim == 1) )
        {
          double num=0; 
          double denom=0;
          for (int d = 0; d<dim; ++d)
          {
            num += normal[d] * p[d] - normal[d] * point_on_plane[d];
            denom += normal[d] * normal[d];
          }
          return num/std::sqrt(denom);
        }
        else
          AssertThrow(false, ExcMessage("Rectangular manifold: dim must be 1, 2 or 3."));
      }

      template <int dim>
      inline double
      rectangular_manifold(const Point<dim> &p,const Point<dim> &lower_left_corner, const Point<dim> &upper_right_corner)
      {
        if (dim == 3)
        {
          std::vector<Point<dim>> n(dim*2);

          for (auto& n_ : n)
            for (int i=0; i<dim; ++i)
              n_[i] = 0;

          n[0][0] = -1.0;
          n[1][1] = -1.0;
          n[2][2] = -1.0;
          n[3][0] = 1.0;
          n[4][1] = 1.0;
          n[5][2] = 1.0;

          double d =1e10;
          for (int i=0; i<dim; ++i)
          {
            double dtemp = signed_distance_planar_manifold<dim>(p, n[i], lower_left_corner );
            d = std::min(d, dtemp);
          }
          for (int i=dim; i<2*dim; ++i)
          {
            double dtemp = signed_distance_planar_manifold<dim>(p, n[i], upper_right_corner );
            d = std::min(d, dtemp);
          }
          return d;
        }
        else if (dim == 2)
        {
          std::vector<Point<dim>> n(dim*2);
          
          for (auto& n_ : n)
            for (int i=0; i<dim; ++i)
              n_[i] = 0;
          
          n[0][0] = -1.0;
          n[1][1] = 1.0;
          n[2][0] = 1.0;
          n[3][1] = -1.0;

          std::vector<Point<dim>> corner(dim*dim);
          std::vector<std::tuple<Point<dim>, Point<dim>>> edges(4);

          corner[0] = lower_left_corner;
          corner[1] = lower_left_corner;
          corner[1][1] = upper_right_corner[1];
          corner[2] = upper_right_corner;
          corner[3] = lower_left_corner;
          corner[3][0] = upper_right_corner[1];
          
          /*
                      
             (1)  +---------------+ (2)
                  |               |
                  |               | 
                  |_______________|
                  |               |
                  |               | 
                  |               |
                  +---------------+
           *    (0)                (3)
           */
          
          if ((p[0]<=corner[0][0]) && (p[1]<=corner[0][1]))
            return p.distance(corner[0]);
          else if ((p[0]<=corner[1][0]) && (p[1]>=corner[1][1]))
            return p.distance(corner[1]);
          else if ((p[0]>=corner[2][0]) && (p[1]>=corner[2][1]))
            return p.distance(corner[2]);
          else if ((p[0]>=corner[3]) && (p[1]<=corner[3][1]))
            return p.distance(corner[3]);
          else if (p[0]<=corner[0][0])
            return = signed_distance_planar_manifold<dim>(p, n[0], lower_left_corner );
          else if (p[1]>=corner[1][1])
            return = signed_distance_planar_manifold<dim>(p, n[1], upper_right_corner );
          else if (p[0]>=corner[2][0])
            return = signed_distance_planar_manifold<dim>(p, n[2], upper_right_corner );
          else if (p[1]<=corner[3][1])
            return = signed_distance_planar_manifold<dim>(p, n[3], lower_left_corner);

          
          if (p[0]<lower_left_corner)

          for (int i=0; i<dim; ++i)
          {
            double dtemp = signed_distance_planar_manifold<dim>(p, n[i], lower_left_corner );
            d = std::min(d, dtemp);
          }
          for (int i=dim; i<2*dim; ++i)
          {
            double dtemp = signed_distance_planar_manifold<dim>(p, n[i], upper_right_corner );
            d = std::min(d, dtemp);
          }
          if( d>0.0)
            std::cout << "d" << d << std::endl;
          return d;
        }
        else
          AssertThrow(false, ExcMessage("Rectangular manifold: dim must be 1, 2 or 3."));
      }



    } // namespace DistanceFunctions

  } // namespace UtilityFunctions
} // namespace MeltPoolDG
