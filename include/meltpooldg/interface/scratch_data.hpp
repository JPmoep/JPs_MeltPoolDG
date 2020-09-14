/* ---------------------------------------------------------------------
 *
 * Author: Peter Münch, Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/

#pragma once
// for parallelization
#include <deal.II/lac/generic_linear_algebra.h>
// enabling conditional ostreams
#include <deal.II/base/conditional_ostream.h> 
// for index set
#include <deal.II/base/index_set.h>
//// for distributed triangulation
#include <deal.II/distributed/tria_base.h>
// for dof_handler type
#include <deal.II/dofs/dof_handler.h>
// for FE_Q<dim> type
#include <deal.II/fe/fe_q.h>
// for mapping
#include <deal.II/fe/mapping.h>

// MeltPoolDG
#include <meltpooldg/interface/simulationbase.hpp>

namespace MeltPoolDG{
  /**
   * Container containing mapping-, finite-element-, and quadrature-related
   * objects to be used either in matrix-based or in matrix-free context.
   */
template <int dim, 
          int spacedim=dim, 
          typename number=double, 
          typename VectorizedArrayType = VectorizedArray<number>>
class ScratchData
{ 
  private:
    using VectorType          = LinearAlgebra::distributed::Vector<double>;    
  public:
    ScratchData()
    {
    }

    /**
     * Setup everything in one go.
     */
    template <int dim_q>
    void
    reinit(
           const Mapping<dim, spacedim> &                        mapping,
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
      this->min_cell_size.emplace_back(GridTools::minimal_cell_diameter(dof_handler.get_triangulation()));
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
      AssertDimension(this->quad_1D.size(), 0);

      this->quad.emplace_back(quadrature);
      return this->quad.size() - 1;
    }

    unsigned int
    attach_quadrature(const Quadrature<1> &quadrature)
    {
      this->quad_1D.emplace_back(quadrature);
      this->quad.emplace_back(QIterated<dim>(quadrature, 1));
      return this->quad.size() - 1;
    }

    void
    build()
    {
      if (this->quad_1D.size() > 0)
        matrix_free.reinit(*this->mapping, this->dof_handler, this->constraint, this->quad_1D);
    }
    /*
     * Store additional more specific data
     */
    //void 
    //set_parameters(const Parameters<number>& parameters)
    //{
      //this->parameters = std::move(parameters);
    //}

    /**
     * Getter functions.
     */
    const Mapping<dim, spacedim> &
    get_mapping() const
    {
      return *this->mapping;
    }

    const FiniteElement<dim, spacedim> &
    get_fe(const unsigned int fe_index=0) const
    {
      return this->dof_handler[fe_index]->get_fe(0);
    }

    const AffineConstraints<number> &
    get_constraint(const unsigned int constraint_index=0) const
    {
      return *this->constraint[constraint_index];
    }

    const Quadrature<dim> &
    get_quadrature(const unsigned int quad_index=0) const
    {
      return this->quad[quad_index];
    }

    const MatrixFree<dim, number, VectorizedArrayType> &
    get_matrix_free() const
    {
      return this->matrix_free;
    }
    
    const DoFHandler<dim, spacedim> &
    get_dof_handler(const unsigned int dof_idx=0) const
    {
      return this->matrix_free.get_dof_handler(dof_idx);
    }


    const double &
    get_min_cell_size(const unsigned int dof_idx=0) const
    {
      return this->min_cell_size[dof_idx];
    }
   
    MPI_Comm  
    get_mpi_comm(const unsigned int dof_idx=0) const
    {
      return UtilityFunctions::get_mpi_comm(*this->dof_handler[dof_idx]);
    }
    
    ConditionalOStream  
    get_pcout(const unsigned int dof_idx=0) const
    {
      return ConditionalOStream( std::cout, Utilities::MPI::this_mpi_process(this->get_mpi_comm(dof_idx)) == 0 );
    }

    void
    initialize_dof_vector(VectorType & vec, const unsigned int dof_idx=0) const
    {
      matrix_free.initialize_dof_vector(vec,dof_idx);
    }

    void
    clear()
    {
      this->matrix_free.clear();
      this->quad_1D.clear();
      this->quad.clear();
      this->constraint.clear();
      this->dof_handler.clear();
      this->mapping.reset();
    }

  private:
    std::unique_ptr<Mapping<dim, spacedim>>        mapping;
    std::vector<const DoFHandler<dim, spacedim> *> dof_handler;
    std::vector<const AffineConstraints<number> *> constraint;
    std::vector<Quadrature<dim>>                   quad;
    std::vector<Quadrature<1>>                     quad_1D;
    std::vector<double>                            min_cell_size;     
    
    MatrixFree<dim, number, VectorizedArrayType>   matrix_free;

};

} // namespace MeltPoolDG
