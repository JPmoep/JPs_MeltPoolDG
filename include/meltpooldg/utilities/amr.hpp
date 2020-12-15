/* ---------------------------------------------------------------------
 *
 * Author:Magdalena Schreter, TUM, September 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

namespace MeltPoolDG
{
  using namespace dealii;

  template <int dim, typename VectorType>
  void
  refine_grid(
    const std::function<bool(parallel::distributed::Triangulation<dim> &)>
      &                                                     mark_cells_for_refinement,
    const std::function<void(std::vector<VectorType *> &)> &attach_vectors,
    const std::function<void()> &                           post,
    const std::function<void(std::shared_ptr<SimulationBase<dim>>, const bool)> &setup_dof_system,
    std::shared_ptr<SimulationBase<dim>> &                                       base_in,
    const DoFHandler<dim> &                                                      dof_handler)
  {
    if (auto tria = std::dynamic_pointer_cast<parallel::distributed::Triangulation<dim>>(
          base_in->triangulation))
      {
        if (mark_cells_for_refinement(*tria) == false)
          return;

        /*
         *  Limit the maximum and minimum refinement levels of cells of the grid.
         */
        if (tria->n_levels() > base_in->parameters.amr.max_grid_refinement_level)
          for (auto &cell : tria->active_cell_iterators_on_level(
                 base_in->parameters.amr.max_grid_refinement_level))
            cell->clear_refine_flag();
        if (tria->n_levels() < base_in->parameters.amr.min_grid_refinement_level)
          for (auto &cell : tria->active_cell_iterators_on_level(
                 base_in->parameters.amr.min_grid_refinement_level))
            cell->clear_coarsen_flag();

        /*
         *  Initialize the triangulation change from the old grid to the new grid
         */
        base_in->triangulation->prepare_coarsening_and_refinement();
        /*
         *  Initialize the solution transfer from the old grid to the new grid
         */
        parallel::distributed::SolutionTransfer<dim, VectorType> solution_transfer(dof_handler);

        std::vector<VectorType *>       new_grid_solutions;
        std::vector<const VectorType *> old_grid_solutions;

        attach_vectors(new_grid_solutions);

        for (const auto &i : new_grid_solutions)
          old_grid_solutions.push_back(i);

        solution_transfer.prepare_for_coarsening_and_refinement(old_grid_solutions);

        /*
         *  Execute the grid refinement
         */
        base_in->triangulation->execute_coarsening_and_refinement();

        /*
         *  update dof-related scratch data to match the current triangulation
         */
        setup_dof_system(base_in, false);

        /*
         *  interpolate the given solution to the new discretization
         *
         */
        solution_transfer.interpolate(new_grid_solutions);

        post();
      }
    else
      //@todo: WIP
      AssertThrow(false, ExcMessage("Mesh refinement for dim=1 not yet supported"));
  }

} // namespace MeltPoolDG
