#pragma once


namespace MeltPoolDG
{
  using namespace dealii;

  namespace VectorTools
  {
    template <int dim, int spacedim, typename Number>
    void
    convert_fe_sytem_vector_to_block_vector(const LinearAlgebra::distributed::Vector<Number> &in,
                                            const DoFHandler<dim, spacedim> &dof_handler_fe_system,
                                            LinearAlgebra::distributed::BlockVector<Number> &out,
                                            const DoFHandler<dim, spacedim> &dof_handler)
    {
      in.update_ghost_values();

      for (const auto &cell_fe_system : dof_handler_fe_system.active_cell_iterators())
        if (cell_fe_system->is_locally_owned())
          {
            Vector<double> local(dof_handler_fe_system.get_fe().n_dofs_per_cell());
            cell_fe_system->get_dof_values(in, local);


            auto cell = DoFCellAccessor<dim, dim, false>(&dof_handler.get_triangulation(),
                                                         cell_fe_system->level(),
                                                         cell_fe_system->index(),
                                                         &dof_handler);

            for (unsigned int d = 0; d < dim; ++d)
              {
                const unsigned int n_dofs_per_component = dof_handler.get_fe().n_dofs_per_cell();
                Vector<double>     local_component(n_dofs_per_component);

                for (unsigned int c = 0; c < n_dofs_per_component; ++c)
                  local_component[c] = local[c * dim + d];

                cell.set_dof_values(local_component, out.block(d));
              }
          }

      in.zero_out_ghosts();
    }

    template <int dim, int spacedim, typename Number>
    void
    convert_block_vector_to_fe_sytem_vector(
      const LinearAlgebra::distributed::BlockVector<Number> &in,
      const DoFHandler<dim, spacedim> &                      dof_handler,
      LinearAlgebra::distributed::Vector<Number> &           out,
      const DoFHandler<dim, spacedim> &                      dof_handler_fe_system)
    {
      in.update_ghost_values();

      for (const auto &cell_fe_system : dof_handler_fe_system.active_cell_iterators())
        if (cell_fe_system->is_locally_owned())
          {
            auto cell = DoFCellAccessor<dim, dim, false>(&dof_handler_fe_system.get_triangulation(),
                                                         cell_fe_system->level(),
                                                         cell_fe_system->index(),
                                                         &dof_handler);

            Vector<double> local(dof_handler_fe_system.get_fe().n_dofs_per_cell());

            for (unsigned int d = 0; d < dim; ++d)
              {
                const unsigned int n_dofs_per_component = dof_handler.get_fe().n_dofs_per_cell();
                Vector<double>     local_component(n_dofs_per_component);

                cell.get_dof_values(in.block(d), local_component);

                for (unsigned int c = 0; c < n_dofs_per_component; ++c)
                  local[c * dim + d] = local_component[c];
              }
            cell_fe_system->set_dof_values(local, out);
          }

      in.zero_out_ghosts();
    }

    template <typename... T>
    void
    update_ghost_values(const T &...args)
    {
      ((args.update_ghost_values()), ...);
    }

    template <typename... T>
    void
    zero_out_ghosts(const T &...args)
    {
      ((args.zero_out_ghosts()), ...);
    }

    template <int dim, typename number>
    static Tensor<1, dim, VectorizedArray<number>>
    normalize(const VectorizedArray<number> &in, const double zero = 1e-16)
    {
      Tensor<1, dim, VectorizedArray<number>> vec;

      for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
        vec[0][v] = in[v] >= zero ? 1.0 : -1.0;

      return vec;
    }

    template <int dim, typename number>
    static Tensor<1, dim, VectorizedArray<number>>
    normalize(const Tensor<1, dim, VectorizedArray<number>> &in, const double zero = 1e-16)
    {
      Tensor<1, dim, VectorizedArray<number>> vec;

      auto n_norm = in.norm();
      for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
        if (n_norm[v] >= zero)
          for (unsigned int d = 0; d < dim; ++d)
            vec[d][v] = in[d][v] / n_norm[v];
        else
          for (unsigned int d = 0; d < dim; ++d)
            vec[d][v] = 0.0;

      return vec;
    }

    template <int dim, typename number = double>
    static Tensor<1, dim, VectorizedArray<number>>
    map_to_vector(const VectorizedArray<number> &in)
    {
      Tensor<1, dim, VectorizedArray<number>> vec;

      for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v)
        vec[0][v] = in[v];

      return vec;
    }

    template <int dim, typename number = double>
    static Tensor<1, dim, VectorizedArray<number>>
    map_to_vector(const Tensor<1, dim, VectorizedArray<number>> &in)
    {
      return in;
    }

  } // namespace VectorTools
} // namespace MeltPoolDG
