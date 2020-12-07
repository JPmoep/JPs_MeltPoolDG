/* ---------------------------------------------------------------------
 *
 * Author: Peter Münch, Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/

#pragma once
// for parallelization
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/partitioner.h>

#include <deal.II/lac/generic_linear_algebra.h>
// for dof_handler type
#include <deal.II/dofs/dof_handler.h>
// for FE_Q<dim> type
#include <deal.II/fe/fe_q.h>
// for mapping
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_tools.h>
// DoFTools
#include <deal.II/dofs/dof_tools.h>

#include <meltpooldg/utilities/fe_integrator.hpp>
#include <meltpooldg/utilities/utilityfunctions.hpp>


namespace MeltPoolDG
{
  /**
   * Container containing mapping-, finite-element-, and quadrature-related
   * objects to be used either in matrix-based or in matrix-free context.
   */
  using namespace dealii;

  template <int dim,
            int spacedim                 = dim,
            typename number              = double,
            typename VectorizedArrayType = VectorizedArray<number>>
  class ScratchData
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<number>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<number>;

  public:
    ScratchData(const bool do_matrix_free = true)
      : do_matrix_free(do_matrix_free)
    {}

    ScratchData(const ScratchData &scratch_data)
    {
      this->do_matrix_free = scratch_data.do_matrix_free;
      this->reinit(scratch_data.get_mapping(),
                   scratch_data.get_dof_handlers(),
                   scratch_data.get_constraints(),
                   scratch_data.get_quadratures());
    }
    /**
     * Setup everything in one go.
     */
    template <int dim_q>
    void
    reinit(const Mapping<dim, spacedim> &                        mapping,
           const std::vector<const DoFHandler<dim, spacedim> *> &dof_handler,
           const std::vector<const AffineConstraints<number> *> &constraint,
           const std::vector<Quadrature<dim_q>> &                quad)
    {
      this->clear();

      this->set_mapping(mapping);

      for (unsigned int i = 0; i < dof_handler.size(); ++i)
        this->attach_dof_handler(*dof_handler[i]);

      for (unsigned int i = 0; i < constraint.size(); ++i)
        this->attach_constraint_matrix(*constraint[i]);

      for (unsigned int i = 0; i < quad.size(); ++i)
        this->attach_quadrature(quad[i]);

      this->create_partitioning();

      this->build();
    }

    /**
     * Fill internal data structures step-by-step.
     */
    void
    set_mapping(const Mapping<dim, spacedim> &mapping)
    {
      this->mapping = mapping.clone();
    }

    unsigned int
    attach_dof_handler(const DoFHandler<dim, spacedim> &dof_handler)
    {
      this->dof_handler.emplace_back(&dof_handler);
      return this->dof_handler.size() - 1;
    }

    unsigned int
    attach_constraint_matrix(const AffineConstraints<number> &constraint)
    {
      this->constraint.emplace_back(&constraint);
      return this->constraint.size() - 1;
    }

    template <int dim_q>
    unsigned int
    attach_quadrature(const Quadrature<dim_q> &quadrature)
    {
      this->quad.emplace_back(Quadrature<dim>(quadrature));
      if constexpr (dim_q == 1) // find a better way to handle face quadrature
        this->face_quad.emplace_back(Quadrature<dim - 1>(quadrature));
      return this->quad.size() - 1;
    }

    void
    create_partitioning()
    {
      /*
       *  recreate DoF-dependent partitioning data
       */
      this->min_cell_size.clear();
      this->diameter.clear();
      this->locally_owned_dofs.clear();
      this->locally_relevant_dofs.clear();
      this->partitioner.clear();

      int dof_idx = 0;
      for (const auto &dof : dof_handler)
        {
          /*
           *  create vector of minimum cell sizes
           */
          this->min_cell_size.push_back(GridTools::minimal_cell_diameter(dof->get_triangulation()) /
                                        std::sqrt(dim));
          /*
           *  create vector of diameters
           */
          this->diameter.push_back(GridTools::minimal_cell_diameter(dof->get_triangulation()));
          // this->diameter.push_back(GridTools::diameter(dof->get_triangulation())); @todo: does
          // not work atm
          /*
           *  create partitioning
           */
          this->locally_owned_dofs.push_back(dof->locally_owned_dofs());

          IndexSet locally_relevant_dofs_temp;
          DoFTools::extract_locally_relevant_dofs(*dof, locally_relevant_dofs_temp);
          this->locally_relevant_dofs.push_back(locally_relevant_dofs_temp);

          this->partitioner.push_back(
            std::make_shared<Utilities::MPI::Partitioner>(this->get_locally_owned_dofs(dof_idx),
                                                          this->get_locally_relevant_dofs(dof_idx),
                                                          this->get_mpi_comm(dof_idx)));
          dof_idx += 1;
        }
    }

    void
    build()
    {
      if (do_matrix_free)
        {
          this->matrix_free.clear();

          typename MatrixFree<dim, double, VectorizedArray<double>>::AdditionalData additional_data;
          additional_data.mapping_update_flags =
            update_values | update_gradients | update_JxW_values | dealii::update_quadrature_points;

          this->matrix_free.reinit(
            *this->mapping, this->dof_handler, this->constraint, this->quad, additional_data);
        }
    }

    /**
     * initialize vectors
     */
    void
    initialize_dof_vector(VectorType &vec, const unsigned int dof_idx = 0) const
    {
      if (do_matrix_free)
        matrix_free.initialize_dof_vector(vec, dof_idx);
      else
        vec.reinit(get_locally_owned_dofs(dof_idx),
                   get_locally_relevant_dofs(dof_idx),
                   get_mpi_comm(dof_idx));
    }

    void
    initialize_dof_vector(BlockVectorType &vec, const unsigned int dof_idx = 0) const
    {
      vec.reinit(dim);
      for (unsigned int d = 0; d < dim; ++d)
        this->initialize_dof_vector(vec.block(d), dof_idx);
    }

    void
    initialize_bc_vector(VectorType &vec, const unsigned int dof_idx = 0) const
    {
      this->initialize_dof_vector(vec, dof_idx);
      this->get_constraint(dof_idx).distribute(vec);
    }

    void
    initialize_bc_vector(BlockVectorType &vec, const unsigned int dof_idx = 0) const
    {
      vec.reinit(dim);
      for (unsigned int d = 0; d < dim; ++d)
        this->initialize_bc_vector(vec.block(d), dof_idx);
    }
    /*
     * clear all member variables
     */
    void
    clear()
    {
      this->matrix_free.clear();
      this->quad.clear();
      this->constraint.clear();
      this->dof_handler.clear();
      this->mapping.reset();
      this->min_cell_size.clear();
      this->locally_owned_dofs.clear();
      this->locally_relevant_dofs.clear();
    }

    /**
     * Getter functions.
     */
    const Mapping<dim, spacedim> &
    get_mapping() const
    {
      return *this->mapping;
    }

    const FiniteElement<dim, spacedim> &
    get_fe(const unsigned int fe_index = 0) const
    {
      return this->dof_handler[fe_index]->get_fe(0);
    }

    const AffineConstraints<number> &
    get_constraint(const unsigned int constraint_index = 0) const
    {
      return *this->constraint[constraint_index];
    }

    const std::vector<const AffineConstraints<number> *> &
    get_constraints() const
    {
      return this->constraint;
    }

    const Quadrature<dim> &
    get_quadrature(const unsigned int quad_index = 0) const
    {
      return this->quad[quad_index];
    }

    const std::vector<Quadrature<dim>> &
    get_quadratures() const
    {
      return this->quad;
    }

    const Quadrature<dim - 1> &
    get_face_quadrature(const unsigned int quad_index = 0) const
    {
      return this->face_quad[quad_index];
    }

    const std::vector<Quadrature<dim - 1>> &
    get_face_quadratures() const
    {
      return this->face_quad;
    }

    const MatrixFree<dim, number, VectorizedArrayType> &
    get_matrix_free() const
    {
      return this->matrix_free;
    }

    const DoFHandler<dim, spacedim> &
    get_dof_handler(const unsigned int dof_idx = 0) const
    {
      return *this->dof_handler[dof_idx];
    }

    const std::vector<const DoFHandler<dim, spacedim> *> &
    get_dof_handlers() const
    {
      return this->dof_handler;
    }

    const Triangulation<dim> &
    get_triangulation(const unsigned int dof_idx = 0) const
    {
      return this->get_dof_handler(dof_idx).get_triangulation();
    }

    unsigned int
    get_n_dofs_per_cell(const unsigned int dof_idx = 0) const
    {
      return get_dof_handler(dof_idx).get_fe().n_dofs_per_cell();
    }

    const double &
    get_min_cell_size(const unsigned int dof_idx = 0) const
    {
      return this->min_cell_size[dof_idx];
    }

    const double &
    get_diameter(const unsigned int dof_idx = 0) const
    {
      return this->diameter[dof_idx];
    }

    const AlignedVector<VectorizedArray<double>> &
    get_cell_diameters(const unsigned int dof_idx      = 0,
                       const unsigned int quad_idx     = 0,
                       bool               do_recompute = false)
    {
      if (do_recompute)
        {
          cell_diameters.resize(this->matrix_free.n_cell_batches());

          FullMatrix<double> mat(dim, dim);

          FECellIntegrator<dim, 1, double> cell_diameter(matrix_free, dof_idx, quad_idx);
          for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches(); ++cell)
            {
              VectorizedArray<double> diameter = VectorizedArray<double>();
              for (unsigned int v = 0; v < VectorizedArray<double>::size(); ++v)
                diameter[v] = this->matrix_free.get_cell_iterator(cell, v, dof_idx)->diameter();
              cell_diameters[cell] = diameter;
            }
        }
      return cell_diameters;
    }

    MPI_Comm
    get_mpi_comm(const unsigned int dof_idx = 0) const
    {
      return UtilityFunctions::get_mpi_comm(*this->dof_handler[dof_idx]);
    }

    const IndexSet &
    get_locally_owned_dofs(const unsigned int dof_idx = 0) const
    {
      return this->locally_owned_dofs[dof_idx];
    }

    const IndexSet &
    get_locally_relevant_dofs(const unsigned int dof_idx = 0) const
    {
      return this->locally_relevant_dofs[dof_idx];
    }

    const std::shared_ptr<Utilities::MPI::Partitioner> &
    get_partitioner(const unsigned int dof_idx = 0) const
    {
      return this->partitioner[dof_idx];
    }

    ConditionalOStream
    get_pcout(const unsigned int dof_idx = 0) const
    {
      return ConditionalOStream(std::cout,
                                Utilities::MPI::this_mpi_process(this->get_mpi_comm(dof_idx)) == 0);
    }

  private:
    bool do_matrix_free;

    std::unique_ptr<Mapping<dim, spacedim>>                   mapping;
    std::vector<const DoFHandler<dim, spacedim> *>            dof_handler;
    std::vector<const AffineConstraints<number> *>            constraint;
    std::vector<Quadrature<dim>>                              quad;
    std::vector<Quadrature<dim - 1>>                          face_quad;
    std::vector<double>                                       min_cell_size;
    std::vector<double>                                       diameter;
    AlignedVector<VectorizedArray<double>>                    cell_diameters;
    std::vector<IndexSet>                                     locally_owned_dofs;
    std::vector<IndexSet>                                     locally_relevant_dofs;
    std::vector<std::shared_ptr<Utilities::MPI::Partitioner>> partitioner;

    MatrixFree<dim, number, VectorizedArrayType> matrix_free;
  };

} // namespace MeltPoolDG
