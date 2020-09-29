/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, TUM, September 2020
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
#include <meltpooldg/interface/scratch_data.hpp>
#include <meltpooldg/interface/problembase.hpp>
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/utilities/timeiterator.hpp>
#include <meltpooldg/reinitialization/reinitialization_operation.hpp>

namespace MeltPoolDG
{
namespace Reinitialization
{
  using namespace dealii; 
 	
  /*
   *     Reinitialization model for reobtaining the "signed-distance" 
   *     property of the level set equation
   */

  template <int dim>
  class ReinitializationProblem : public ProblemBase<dim>
  {
  private:
    using VectorType          = LinearAlgebra::distributed::Vector<double>;         
    using DoFHandlerType      = DoFHandler<dim>;                                    

  public:

    /*
     *  Constructor of reinitialization problem
     */

    ReinitializationProblem() = default;

    void 
    run( std::shared_ptr<SimulationBase<dim>> base_in ) final
    {
      initialize(base_in);

      while ( !time_iterator.is_finished() )
      {
        const double d_tau = time_iterator.get_next_time_increment();   
        scratch_data->get_pcout() << "t= " << std::setw(10) << std::left << time_iterator.get_current_time();
        
        reinit_operation.solve(d_tau);
        
        output_results(time_iterator.get_current_time_step_number(),
                       base_in->parameters);
      }
    }

    std::string get_name() final { return "reinitialization"; };

  private:
    /*
     *  This function initials the relevant member data
     *  for the computation of a reinitialization problem
     */
    void 
    initialize( std::shared_ptr<SimulationBase<dim>> base_in )
    {
      /*
       *  setup scratch data
       */
      scratch_data = std::make_shared<ScratchData<dim>>();
      /*
       *  setup mapping
       */
      auto mapping = MappingQGeneric<dim>(base_in->parameters.base.degree);
      scratch_data->set_mapping(mapping);
      /*
       *  setup DoFHandler
       */
      FE_Q<dim>    fe(base_in->parameters.base.degree);
      
      dof_handler.initialize(*base_in->triangulation, fe );
      scratch_data->attach_dof_handler(dof_handler);

      /*
       *  make hanging nodes constraints
       */
      constraints.clear();
      constraints.reinit(scratch_data->get_locally_relevant_dofs());
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      constraints.close();
      
      scratch_data->attach_constraint_matrix(constraints);
      /*
       *  create quadrature rule
       */
      QGauss<1> quad_1d_temp(base_in->parameters.base.n_q_points_1d);
      
      scratch_data->attach_quadrature(quad_1d_temp);
      /*
       *  create the matrix-free object
       */
      scratch_data->build();

      /*  
       *  initialize the time iterator
       */
      time_iterator.initialize(TimeIteratorData<double>{ 0.0,
                                                 10000.,
                                                 base_in->parameters.reinit.dtau,
                                                 base_in->parameters.reinit.max_n_steps,
                                                 false });
      /*
       *  set initial conditions of the levelset function
       */
      VectorType solution_level_set;
      scratch_data->initialize_dof_vector(solution_level_set);
      VectorTools::project( dof_handler, 
                            constraints,
                            scratch_data->get_quadrature(),
                            *base_in->get_field_conditions()->initial_field,           
                            solution_level_set );

      solution_level_set.update_ghost_values();

      /*
       *    initialize the reinitialization operation class
       */
      reinit_operation.initialize(scratch_data, solution_level_set, base_in->parameters);
    }

    /*
     *  Creating paraview output
     */
    void 
    output_results(const unsigned int time_step,
                   const Parameters<double>& parameters) const
    {
      if (parameters.paraview.do_output)
      {
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(reinit_operation.solution_level_set, "psi");

        if (parameters.paraview.print_normal_vector)
        {
          for (unsigned int d=0; d<dim; ++d)
            data_out.add_data_vector(reinit_operation.solution_normal_vector.block(d), "normal_"+std::to_string(d));
        }

        const int n_digits_timestep = 4;
        const int n_groups = 1;
        data_out.build_patches();
        data_out.write_vtu_with_pvtu_record("./", parameters.paraview.filename, time_step, scratch_data->get_mpi_comm(), n_digits_timestep, n_groups);
      }
    }

  private:
    DoFHandler<dim>                    dof_handler;
    AffineConstraints<double>          constraints;
    
    std::shared_ptr<ScratchData<dim>>  scratch_data;
    TimeIterator<double>               time_iterator;
    ReinitializationOperation<dim>     reinit_operation;
    
  };
} // namespace Reinitialization
} // namespace MeltPoolDG
