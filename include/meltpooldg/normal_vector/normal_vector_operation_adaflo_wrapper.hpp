/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
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
#include <meltpooldg/normal_vector/normal_vector_operation_base.hpp>
#include <meltpooldg/utilities/utilityfunctions.hpp>

#include <adaflo/block_matrix_extension.h>
#include <adaflo/level_set_okz_compute_normal.h>
#include <adaflo/level_set_okz_preconditioner.h>
#include <adaflo/util.h>

namespace MeltPoolDG
{
  namespace NormalVector
  {
    using namespace dealii;

    template <int dim>
    class NormalVectorOperationAdaflo : public NormalVectorOperationBase<dim>
    {
    private:
      using VectorType       = LinearAlgebra::distributed::Vector<double>;
      using BlockVectorType  = LinearAlgebra::distributed::BlockVector<double>;
      using SparseMatrixType = TrilinosWrappers::SparseMatrix;

    public:
      /**
       * Constructor.
       */
      NormalVectorOperationAdaflo(const ScratchData<dim> &  scratch_data,
                                  const int                 advec_diff_dof_idx,
                                  const int                 normal_vec_dof_idx,
                                  const int                 normal_vec_quad_idx,
                                  const VectorType &        advected_field, //@todo: make const
                                  const Parameters<double> &data_in)
        : scratch_data(scratch_data)
        , advec_diff_dof_idx(advec_diff_dof_idx)
        , normal_vec_dof_idx(normal_vec_dof_idx)
        , normal_vec_quad_idx(normal_vec_quad_idx)
        , normal_vector_field(dim)
        , rhs(dim)
      {
        /**
         * set parameters of adaflo
         */
        set_adaflo_parameters(data_in, advec_diff_dof_idx, normal_vec_dof_idx, normal_vec_quad_idx);

        /**
         * initialize the projection matrix
         */
        projection_matrix     = std::make_shared<BlockMatrixExtension>();
        ilu_projection_matrix = std::make_shared<BlockILUExtension>();

        /*
         * initialize adaflo operation
         */
        normal_vec_operation =
          std::make_shared<LevelSetOKZSolverComputeNormal<dim>>(normal_vector_field,
                                                                rhs,
                                                                advected_field,
                                                                scratch_data.get_cell_diameters(),
                                                                cell_diameter_max, // @todo
                                                                cell_diameter_min,
                                                                scratch_data.get_constraint(
                                                                  normal_vec_dof_idx),
                                                                normal_vec_adaflo_params,
                                                                scratch_data.get_matrix_free(),
                                                                preconditioner,
                                                                projection_matrix,
                                                                ilu_projection_matrix);

        reinit();
      }

      void
      reinit() override
      {
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
                                                       normal_vec_dof_idx),
                                                     normal_vec_dof_idx,
                                                     normal_vec_quad_idx,
                                                     preconditioner);


        initialize_projection_matrix<dim, double, VectorizedArray<double>>(
          scratch_data.get_matrix_free(),
          scratch_data.get_constraint(normal_vec_dof_idx),
          normal_vec_dof_idx,
          normal_vec_quad_idx,
          cell_diameter_max, // @todo
          cell_diameter_min, // @todo
          scratch_data.get_cell_diameters(),
          *projection_matrix,
          *ilu_projection_matrix);
      }

      /**
       * Solver time step
       */
      void
      solve(const VectorType &advected_field) override
      {
        (void)advected_field;
        initialize_vectors();
        normal_vec_operation->compute_normal(true);

        for (unsigned int d = 0; d < dim; ++d)
          scratch_data.get_pcout() << " |n|_" << d << "=" << std::setw(15) << std::setprecision(10)
                                   << std::left << get_solution_normal_vector().block(d).l2_norm();

        scratch_data.get_pcout() << std::endl;
      }

      /**
       *  getter
       */
      const LinearAlgebra::distributed::BlockVector<double> &
      get_solution_normal_vector() const override
      {
        return normal_vector_field;
      }

      LevelSetOKZSolverComputeNormal<dim> &
      get_adaflo_obj()
      {
        return *normal_vec_operation;
      }

    private:
      void
      set_adaflo_parameters(const Parameters<double> &parameters,
                            const int                 advec_diff_dof_idx,
                            const int                 normal_vec_dof_idx,
                            const int                 normal_vec_quad_idx)
      {
        normal_vec_adaflo_params.dof_index_ls         = advec_diff_dof_idx;
        normal_vec_adaflo_params.dof_index_normal     = normal_vec_dof_idx;
        normal_vec_adaflo_params.quad_index           = normal_vec_quad_idx;
        normal_vec_adaflo_params.damping_scale_factor = parameters.normal_vec.damping_scale_factor;
        normal_vec_adaflo_params.epsilon              = 1.0;      //@ todo
        normal_vec_adaflo_params.approximate_projections = false; //@ todo
      }

      void
      initialize_vectors()
      {
        /**
         * initialize advected field dof vectors
         */
        scratch_data.initialize_dof_vector(normal_vector_field,
                                           normal_vec_adaflo_params.dof_index_normal);
        /**
         * initialize vectors for the solution of the linear system
         */
        scratch_data.initialize_dof_vector(rhs, normal_vec_adaflo_params.dof_index_normal);
      }

    private:
      const ScratchData<dim> &scratch_data;
      const unsigned int      advec_diff_dof_idx;
      const unsigned int      normal_vec_dof_idx;
      const unsigned int      normal_vec_quad_idx;
      /**
       *  Vectors for computing the normals
       */
      BlockVectorType normal_vector_field;
      BlockVectorType rhs;
      /**
       * Adaflo parameters for the level set problem
       */
      LevelSetOKZSolverComputeNormalParameter normal_vec_adaflo_params;
      /**
       * Reference to the actual advection diffusion solver from adaflo
       */
      std::shared_ptr<LevelSetOKZSolverComputeNormal<dim>> normal_vec_operation;

      /**
       *  Diagonal preconditioner @todo
       */
      DiagonalPreconditioner<double>         preconditioner;
      std::shared_ptr<BlockMatrixExtension>  projection_matrix;
      std::shared_ptr<BlockILUExtension>     ilu_projection_matrix;
      AlignedVector<VectorizedArray<double>> cell_diameters;
      double                                 cell_diameter_min;
      double                                 cell_diameter_max;
    };
  } // namespace NormalVector
} // namespace MeltPoolDG
