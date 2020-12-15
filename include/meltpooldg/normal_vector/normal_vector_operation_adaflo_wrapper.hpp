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
#include <meltpooldg/normal_vector/normal_vector_operation_base.hpp>

#include <meltpooldg/interface/operator_base.hpp>
#include <meltpooldg/interface/parameters.hpp>
#include <meltpooldg/utilities/utilityfunctions.hpp>

#include <adaflo/level_set_okz_compute_normal.h>
#include <adaflo/block_matrix_extension.h>

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
      NormalVectorOperationAdaflo(const ScratchData<dim> & scratch_data,
                    const int                            advec_diff_dof_idx,
                    const int                            normal_vec_dof_idx,
                    const int                            normal_vec_quad_idx,
                    const VectorType                     advected_field,
                    const Parameters<double>&            data_in)
        : scratch_data(scratch_data)
      {
        /**
         * set parameters of adaflo
         */
        set_adaflo_parameters(data_in,
                              advec_diff_dof_idx,
                              normal_vec_dof_idx,
                              normal_vec_quad_idx);
        /**
         *  initialize the dof vectors
         */
        initialize_vectors();
        /*
         * initialize adaflo operation
         */
        normal_vec_operation = std::make_shared<LevelSetOKZSolverComputeNormal<dim>>(
          normal_vector_field,
          rhs,
          solution_temp,
          scratch_data.get_cell_diameters(),
          normal_vec_adaflo_params.epsilon, // @todo
          0.0 /*minimal edge length*/,      // @todo
          scratch_data.get_constraint(normal_vec_dof_idx),
          normal_vec_adaflo_params,
          scratch_data.get_matrix_free(),
          preconditioner,
          projection_matrix,
          ilu_projection_matrix
          );
        /**
         * initialize the preconditioner
         */
        //initialize_mass_matrix_diagonal<dim, double>(scratch_data.get_matrix_free(),
                                                     //scratch_data.get_constraint(
                                                       //advec_diff_dof_idx),
                                                     //advec_diff_dof_idx,
                                                     //advec_diff_quad_idx,
                                                     //preconditioner);
      }

      /**
       * Solver time step
       */
      void
      solve(const VectorType &advected_field) override
      {
        (void)advected_field;
        //advec_diff_operation->advance_concentration(dt);

        //scratch_data.get_pcout() << " |phi|= " << std::setw(15) << std::setprecision(10)
                                 //<< std::left << get_advected_field().l2_norm()
                                 //<< " |phi_n-1|= " << std::setw(15) << std::setprecision(10)
                                 //<< std::left << get_advected_field_old().l2_norm()
                                 //<< " |phi_n-2|= " << std::setw(15) << std::setprecision(10)
                                 //<< std::left << get_advected_field_old_old().l2_norm()
                                 //<< std::endl;
      }

      const LinearAlgebra::distributed::BlockVector<double> &
      get_solution_normal_vector() const override
      {
        return normal_vector_field;
      }

    private:
      void
      set_adaflo_parameters(const Parameters<double> &parameters,
                            const int                 advec_diff_dof_idx,
                            const int                 normal_vec_dof_idx,
                            const int                 normal_vec_quad_idx)
      {
        normal_vec_adaflo_params.dof_index_ls            = advec_diff_dof_idx;
        normal_vec_adaflo_params.dof_index_normal        = normal_vec_dof_idx;
        normal_vec_adaflo_params.quad_index              = normal_vec_quad_idx;
        normal_vec_adaflo_params.epsilon                 = 0.0; //@ todo
        normal_vec_adaflo_params.approximate_projections = false; //@ todo
      }

      void
      initialize_vectors()
      {
        /**
         * initialize advected field dof vectors
         */
        //scratch_data.initialize_dof_vector(advected_field, adaflo_params.dof_index_ls);
        //scratch_data.initialize_dof_vector(advected_field_old, adaflo_params.dof_index_ls);
        //scratch_data.initialize_dof_vector(advected_field_old_old, adaflo_params.dof_index_ls);
        /**
         * initialize vectors for the solution of the linear system
         */
        //scratch_data.initialize_dof_vector(rhs, adaflo_params.dof_index_ls);
        //scratch_data.initialize_dof_vector(increment, adaflo_params.dof_index_ls);
        /**
         *  initialize the velocity vector for adaflo
         */
        //scratch_data.initialize_dof_vector(velocity_vec, adaflo_params.dof_index_vel);
        //scratch_data.initialize_dof_vector(velocity_vec_old, adaflo_params.dof_index_vel);
        //scratch_data.initialize_dof_vector(velocity_vec_old_old, adaflo_params.dof_index_vel);
      }

    private:
      const ScratchData<dim> &scratch_data;
      /**
       *  Vectors for computing the normals 
       */
      BlockVectorType normal_vector_field;
      BlockVectorType rhs;
      VectorType solution_temp;
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
      DiagonalPreconditioner<double>        preconditioner;
      std::shared_ptr<BlockMatrixExtension> projection_matrix;
      std::shared_ptr<BlockILUExtension>    ilu_projection_matrix;
    };
  } // namespace NormalVector
} // namespace MeltPoolDG
