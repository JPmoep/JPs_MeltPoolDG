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
  refine_grid(const std::function<bool(parallel::distributed::Triangulation<dim> &)>
                &mark_cells_for_refinement,
              const std::vector<std::pair<const DoFHandler<dim> *,
                                          std::function<void(std::vector<VectorType *> &)>>> &data,
              const std::function<void()> &                                                   post,
              const std::function<void()> &setup_dof_system,
              const AdaptiveMeshingData &  amr)
  {
    const unsigned int n = data.size();

    Assert(n > 0, ExcNotImplemented());

    auto triangulation = const_cast<Triangulation<dim> *>(&data[0].first->get_triangulation());

    Assert(triangulation, ExcNotImplemented());

    if (auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> *>(triangulation))
      {
        if (mark_cells_for_refinement(*tria) == false)
          return;

        /*
         *  Limit the maximum and minimum refinement levels of cells of the grid.
         */
        if (tria->n_levels() > amr.max_grid_refinement_level)
          for (auto &cell : tria->active_cell_iterators_on_level(amr.max_grid_refinement_level))
            cell->clear_refine_flag();
        if (tria->n_levels() < amr.min_grid_refinement_level)
          for (auto &cell : tria->active_cell_iterators_on_level(amr.min_grid_refinement_level))
            cell->clear_coarsen_flag();

        /*
         *  Initialize the triangulation change from the old grid to the new grid
         */
        tria->prepare_coarsening_and_refinement();
        /*
         *  Initialize the solution transfer from the old grid to the new grid
         */
        std::vector<std::shared_ptr<parallel::distributed::SolutionTransfer<dim, VectorType>>>
          solution_transfer(n);

        std::vector<std::vector<VectorType *>>       new_grid_solutions(n);
        std::vector<std::vector<const VectorType *>> old_grid_solutions(n);

        for (unsigned int j = 0; j < n; ++j)
          {
            data[j].second(new_grid_solutions[j]);

            for (const auto &i : new_grid_solutions[j])
              old_grid_solutions[j].push_back(i);

            solution_transfer[j] =
              std::make_shared<parallel::distributed::SolutionTransfer<dim, VectorType>>(
                *data[j].first);
            solution_transfer[j]->prepare_for_coarsening_and_refinement(old_grid_solutions[j]);
          }
        /*
         *  Execute the grid refinement
         */
        tria->execute_coarsening_and_refinement();

        /*
         *  update dof-related scratch data to match the current triangulation
         */
        setup_dof_system();

        /*
         *  interpolate the given solution to the new discretization
         *
         */
        for (unsigned int j = 0; j < n; ++j)
          solution_transfer[j]->interpolate(new_grid_solutions[j]);
        post();
      }
    else
      //@todo: WIP
      AssertThrow(false, ExcMessage("Mesh refinement for dim=1 not yet supported"));
  }


  template <int dim, typename VectorType>
  void
  refine_grid(const std::function<bool(parallel::distributed::Triangulation<dim> &)>
                &                                                     mark_cells_for_refinement,
              const std::function<void(std::vector<VectorType *> &)> &attach_vectors,
              const std::function<void()> &                           post,
              const std::function<void()> &                           setup_dof_system,
              const AdaptiveMeshingData &                             amr,
              const DoFHandler<dim> &                                 dof_handler)
  {
    refine_grid<dim, VectorType>(
      mark_cells_for_refinement,
      {std::pair<const DoFHandler<dim> *, std::function<void(std::vector<VectorType *> &)>>{
        &dof_handler, attach_vectors}},
      post,
      setup_dof_system,
      amr);
  }

} // namespace MeltPoolDG
