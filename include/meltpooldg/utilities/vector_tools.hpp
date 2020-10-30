#pragma once


namespace MeltPoolDG
{
using namespace dealii;

    namespace
    VectorTools
    {
      template<int dim, int spacedim, typename Number>
      void
      convert_fe_sytem_vector_to_block_vector(const LinearAlgebra::distributed::Vector<Number>& in, const DoFHandler<dim, spacedim> & dof_handler_adaflo, LinearAlgebra::distributed::BlockVector<Number>& out, const DoFHandler<dim, spacedim> & dof_handler)
      {
        in.update_ghost_values();
          
        for (const auto &cell_adaflo : dof_handler_adaflo.active_cell_iterators())
          if (cell_adaflo->is_locally_owned())
          {
              Vector<double> local(dof_handler_adaflo.get_fe().n_dofs_per_cell());
              cell_adaflo->get_dof_values(in, local);


              auto cell = DoFCellAccessor<dim, dim, false>(&dof_handler.get_triangulation(),
                                              cell_adaflo->level(), 
                                              cell_adaflo->index(),   
                                             &dof_handler);
              
              for (unsigned int d=0; d<dim; ++d)
              {
                const unsigned int n_dofs_per_component = dof_handler.get_fe().n_dofs_per_cell();
                Vector<double> local_component(n_dofs_per_component);

                for(unsigned int c = 0; c < n_dofs_per_component; ++c)
                  local_component[c] = local[c * dim + d];

                 cell.set_dof_values(local_component, out.block(d));
              }
          }

        in.zero_out_ghosts();
      }

      template<int dim, int spacedim, typename Number>
      void
      convert_block_vector_to_fe_sytem_vector(const LinearAlgebra::distributed::BlockVector<Number>& in, const DoFHandler<dim, spacedim> & dof_handler, LinearAlgebra::distributed::Vector<Number>& out, const DoFHandler<dim, spacedim> & dof_handler_adaflo)
      {
        in.update_ghost_values();
        
        for (const auto &cell_adaflo : dof_handler_adaflo.active_cell_iterators())
          if (cell_adaflo->is_locally_owned())
          {
              auto cell = DoFCellAccessor<dim, dim, false>(&dof_handler.get_triangulation(),
                                              cell_adaflo->level(), 
                                              cell_adaflo->index(),   
                                             &dof_handler);

              Vector<double> local(dof_handler_adaflo.get_fe().n_dofs_per_cell());
              
              for (unsigned int d=0; d<dim; ++d)
              {
                 const unsigned int n_dofs_per_component = dof_handler.get_fe().n_dofs_per_cell();
                 Vector<double> local_component(n_dofs_per_component);

                 cell.get_dof_values(in.block(d), local_component);

                 for(unsigned int c = 0; c < n_dofs_per_component; ++c)
                   local[c * dim + d] = local_component[c];
              }
              cell_adaflo->set_dof_values(local, out);
          }

        in.zero_out_ghosts();
      }
    }

} // namespace MeltPoolDG
