/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, December 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
// DoFTools
#include <deal.II/dofs/dof_tools.h>
// MeltPoolDG
#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/curvature/curvature_operation_base.hpp>
#include <meltpooldg/utilities/utilityfunctions.hpp>
#include <meltpooldg/normal_vector/normal_vector_operation_adaflo_wrapper.hpp>

#include <adaflo/block_matrix_extension.h>
#include <adaflo/level_set_okz_compute_curvature.h>
#include <adaflo/level_set_okz_preconditioner.h>
#include <adaflo/util.h>

namespace MeltPoolDG
{
  namespace Curvature
  {
    using namespace dealii;

    template <int dim>
    class CurvatureOperationAdaflo : public CurvatureOperationBase<dim>
    {
    private:
      using VectorType       = LinearAlgebra::distributed::Vector<double>;
      using BlockVectorType  = LinearAlgebra::distributed::BlockVector<double>;
      using SparseMatrixType = TrilinosWrappers::SparseMatrix;

    public:
      /**
       * Constructor.
       */
      CurvatureOperationAdaflo(const ScratchData<dim> &  scratch_data,
                                  const int                 advec_diff_dof_idx,
                                  const int                 normal_vec_dof_idx,
                                  const int                 curv_dof_idx,
                                  const int                 curv_quad_idx,
                                  VectorType &              advected_field, //@todo: make const
                                  const Parameters<double> &data_in)
        : scratch_data(scratch_data)
        , normal_vector_operation(  // @todo: this should be replaced by the melt pool normal vector operation (?)?
            scratch_data,
            advec_diff_dof_idx,
            normal_vec_dof_idx,
            curv_quad_idx,
            advected_field, //@todo: make const
            data_in)
      {
        /**
         * set parameters of adaflo
         */
        set_adaflo_parameters(data_in, advec_diff_dof_idx, curv_dof_idx, curv_quad_idx);
        /**
         *  initialize the dof vectors
         */
        initialize_vectors();

        compute_cell_diameters<dim>(scratch_data.get_matrix_free(),
                                    advec_diff_dof_idx,
                                    cell_diameters,
                                    cell_diameter_min,
                                    cell_diameter_max);

        /**
         * initialize the preconditioner -->  @todo: currently not used in adaflo
         */
        initialize_mass_matrix_diagonal<dim, double>(scratch_data.get_matrix_free(),
                                                     scratch_data.get_constraint(
                                                       curv_dof_idx),
                                                     curv_dof_idx,
                                                     curv_quad_idx,
                                                     preconditioner);

        /**
         * initialize the projection matrix
         */
        projection_matrix     = std::make_shared<BlockMatrixExtension>();
        ilu_projection_matrix = std::make_shared<BlockILUExtension>();

        initialize_projection_matrix<dim, double, VectorizedArray<double>>(
          scratch_data.get_matrix_free(),
          scratch_data.get_constraint(curv_dof_idx),
          curv_dof_idx,
          curv_quad_idx,
          cell_diameter_max, // @todo
          cell_diameter_min, // @todo
          scratch_data.get_cell_diameters(),
          *projection_matrix,
          *ilu_projection_matrix);
        /*
         * initialize adaflo operation for computing curvature
         */
        curvature_operation =
          std::make_shared<LevelSetOKZSolverComputeCurvature<dim>>(normal_vector_operation.get_adaflo_obj(), // @todo: get rid of this function argument in adaflo
                                                                scratch_data.get_cell_diameters(),
                                                                normal_vector_operation.get_solution_normal_vector(),
                                                                scratch_data.get_constraint(curv_dof_idx),
                                                                scratch_data.get_constraint(curv_dof_idx), // @todo -- check adaflo --> hanging node constraints??
                                                                cell_diameter_max, // @todo
                                                                rhs,
                                                                curv_adaflo_params,
                                                                curvature_field,
                                                                advected_field,
                                                                scratch_data.get_matrix_free(),
                                                                preconditioner,
                                                                projection_matrix,
                                                                ilu_projection_matrix);
      }

      /**
       * Solver time step
       */
      void
      solve(const VectorType &advected_field) override
      {
        (void)advected_field;
        initialize_vectors();
        curvature_operation->compute_curvature(true); // @todo: adaflo does not use the boolean function argument

        scratch_data.get_pcout() << " |k|=" << std::setw(15) << std::setprecision(10)
                                   << std::left << get_curvature().l2_norm();

        scratch_data.get_pcout() << std::endl;
      }

      const LinearAlgebra::distributed::Vector<double> &
      get_curvature() const override
      {
        return curvature_field;
      }
      
      LinearAlgebra::distributed::Vector<double> &
      get_curvature() override
      {
        return curvature_field;
      }
      
      const LinearAlgebra::distributed::BlockVector<double> &
      get_normal_vector() const override
      {
        // @ todo
      }

    private:
      void
      set_adaflo_parameters(const Parameters<double> &parameters,
                            const int                 advec_diff_dof_idx,
                            const int                 curv_dof_idx,
                            const int                 curv_quad_idx)
      {
        curv_adaflo_params.dof_index_ls            = advec_diff_dof_idx;
        curv_adaflo_params.dof_index_curvature     = curv_dof_idx; //@ todo
        curv_adaflo_params.dof_index_normal        = curv_dof_idx;
        curv_adaflo_params.quad_index              = curv_quad_idx;
        curv_adaflo_params.epsilon                 = 1.0;      //@ todo
        curv_adaflo_params.approximate_projections = false; //@ todo
        curv_adaflo_params.curvature_correction    = false; //@ todo
        //curv_adaflo_params.damping_scale_factor = parameters.normal_vec.damping_scale_factor;
      }

      void
      initialize_vectors()
      {
        /**
         * initialize advected field dof vectors
         */
        scratch_data.initialize_dof_vector(curvature_field,
                                           curv_adaflo_params.dof_index_curvature);
        /**
         * initialize vectors for the solution of the linear system
         */
        scratch_data.initialize_dof_vector(rhs, curv_adaflo_params.dof_index_curvature);
      }

    private:
      const ScratchData<dim> &scratch_data;
      /**
       *  Vectors for computing the normals
       */
      VectorType curvature_field;
      VectorType rhs;
      /**
       * Adaflo parameters for the curvature problem
       */
      LevelSetOKZSolverComputeCurvatureParameter curv_adaflo_params;
      /**
       * Reference to the actual curvature solver from adaflo
       */
      std::shared_ptr<LevelSetOKZSolverComputeCurvature<dim>> curvature_operation;

      /**
       *  Diagonal preconditioner
       */
      DiagonalPreconditioner<double>                 preconditioner;
      /**
       *  Projection matrices
       */
      std::shared_ptr<BlockMatrixExtension>          projection_matrix;
      std::shared_ptr<BlockILUExtension>             ilu_projection_matrix;
      /**
       *  Geometry info
       */
      AlignedVector<VectorizedArray<double>>         cell_diameters;
      double                                         cell_diameter_min;
      double                                         cell_diameter_max;
      /**
       *  Adaflo normal vector operation wrapper @todo: should be replaced by generic normal_vector_operation_base
       */
      NormalVector::NormalVectorOperationAdaflo<dim> normal_vector_operation;
    };
  } // namespace Curvature
} // namespace MeltPoolDG
