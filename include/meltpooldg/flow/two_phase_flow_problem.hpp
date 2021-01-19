/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, Peter Münch, TUM, October 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/index_set.h>

#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/generic_linear_algebra.h>
// MeltPoolDG
#include <meltpooldg/flow/adaflo_wrapper.hpp>
#include <meltpooldg/flow/flow_base.hpp>
#include <meltpooldg/interface/problembase.hpp>
#include <meltpooldg/interface/simulationbase.hpp>
#include <meltpooldg/level_set/level_set_operation.hpp>
#include <meltpooldg/melt_pool/melt_pool_operation.hpp>
#include <meltpooldg/utilities/amr.hpp>
#include <meltpooldg/utilities/timeiterator.hpp>
#include <meltpooldg/utilities/vector_tools.hpp>

namespace MeltPoolDG::Flow
{
  using namespace dealii;

  template <int dim>
  class TwoPhaseFlowProblem : public ProblemBase<dim>
  {
  private:
    using VectorType      = LinearAlgebra::distributed::Vector<double>;
    using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

  public:
    TwoPhaseFlowProblem() = default;

    void
    run(std::shared_ptr<SimulationBase<dim>> base_in) final
    {
      initialize(base_in);

      while (!time_iterator.is_finished())
        {
          const auto dt = time_iterator.get_next_time_increment();
          const auto n  = time_iterator.get_current_time_step_number();

          scratch_data->get_pcout()
            << "t= " << std::setw(10) << std::left << time_iterator.get_current_time();

          // ... solve level-set problem with the given advection field
          level_set_operation.solve(dt, flow_operation->get_velocity());

          // update
          update_phases(level_set_operation.level_set_as_heaviside, base_in->parameters);

          // accumulate forces: a) gravity force
          compute_gravity_force(force_rhs, base_in->parameters.base.gravity);

          // ... b) surface tension
          if (base_in->parameters.flow.temperature_dependent_surface_tension_coefficient == 0)
            level_set_operation.compute_surface_tension(
              force_rhs,
              base_in->parameters.flow.surface_tension_coefficient,
              vel_dof_idx,
              flow_quad_idx,
              false /*false means add to force vector*/);

          if (base_in->parameters.base.problem_name == "melt_pool")
            {
              // ... c) recoil pressure (+ compute temperature from analytical field)
              melt_pool_operation.compute_recoil_pressure_force(
                force_rhs,
                level_set_operation.level_set_as_heaviside,
                dt,
                false /*false means add to force vector*/);

              //// ... d) temperature-dependent surface tension
              if (base_in->parameters.flow.temperature_dependent_surface_tension_coefficient > 0.0)
                melt_pool_operation.compute_temperature_dependent_surface_tension(
                  force_rhs,
                  level_set_operation.level_set_as_heaviside,
                  level_set_operation.get_curvature(),
                  base_in->parameters.flow.surface_tension_coefficient,
                  base_in->parameters.flow.temperature_dependent_surface_tension_coefficient,
                  base_in->parameters.flow.surface_tension_reference_temperature,
                  ls_dof_idx,
                  vel_dof_idx,
                  flow_quad_idx,
                  temp_dof_idx,
                  false /*false means add to force vector*/);

              if (base_in->parameters.mp.set_velocity_to_zero_in_solid)
                {
                  // set the fluid velocity in solid regions to zero
#ifdef MELT_POOL_DG_WITH_ADAFLO
                  melt_pool_operation.set_flow_field_in_solid_regions_to_zero(
                    flow_operation->get_dof_handler_velocity(),
                    flow_operation->get_constraints_velocity());
#else
                  AssertThrow(false, ExcNotImplemented());
#endif
                }
            }

          //  ... and set forces within the Navier-Stokes solver
          flow_operation->set_force_rhs(force_rhs);

          // solver Navier-Stokes problem
          flow_operation->solve();

          // ... and output the results to vtk files.
          output_results(n, base_in->parameters.base.problem_name == "melt_pool");

          if (base_in->parameters.amr.do_amr)
            refine_mesh(base_in);
        }
    }

    std::string
    get_name() final
    {
      return "two_phase_flow";
    };

  private:
    /*
     *  This function initials the relevant scratch data
     *  for the computation of the level set problem
     */
    void
    initialize(std::shared_ptr<SimulationBase<dim>> base_in)
    {
      /*
       *  setup DoFHandler
       */
      dof_handler.reinit(*base_in->triangulation);
      flow_dof_handler.reinit(*base_in->triangulation);

      /*
       *  setup scratch data
       */
      {
        scratch_data = std::make_shared<ScratchData<dim>>(/*do_matrix_free*/ true);

        /*
         *  setup mapping
         */
#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
        if (base_in->parameters.base.do_simplex)
          scratch_data->set_mapping(
            MappingFE<dim>(Simplex::FE_P<dim>(base_in->parameters.base.degree)));
        else
#endif
          scratch_data->set_mapping(MappingQGeneric<dim>(base_in->parameters.base.degree));

        scratch_data->attach_dof_handler(dof_handler);
        scratch_data->attach_dof_handler(dof_handler);
        scratch_data->attach_dof_handler(dof_handler);
        scratch_data->attach_dof_handler(flow_dof_handler);

        ls_hanging_nodes_dof_idx =
          scratch_data->attach_constraint_matrix(ls_hanging_node_constraints);
        ls_dof_idx     = scratch_data->attach_constraint_matrix(ls_constraints_dirichlet);
        reinit_dof_idx = scratch_data->attach_constraint_matrix(reinit_constraints_dirichlet);
        flow_dof_idx   = scratch_data->attach_constraint_matrix(flow_dummy_constraint);

        /*
         *  create quadrature rule
         */
#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
        if (base_in->parameters.base.do_simplex)
          {
            ls_quad_idx = scratch_data->attach_quadrature(
              Simplex::QGauss<dim>(base_in->parameters.base.n_q_points_1d));
            flow_quad_idx = scratch_data->attach_quadrature(
              Simplex::QGauss<dim>(base_in->parameters.flow.velocity_degree + 1));
          }
        else
#endif
          {
            ls_quad_idx =
              scratch_data->attach_quadrature(QGauss<1>(base_in->parameters.base.n_q_points_1d));
            flow_quad_idx = scratch_data->attach_quadrature(
              QGauss<1>(base_in->parameters.flow.velocity_degree + 1));
          }
      }

#ifdef MELT_POOL_DG_WITH_ADAFLO
      flow_operation = std::make_shared<AdafloWrapper<dim>>(*scratch_data, base_in);
#else
      AssertThrow(false, ExcNotImplemented());
#endif
      /*
       *  set indices of flow dof handlers
       */
      vel_dof_idx      = flow_operation->get_dof_handler_idx_velocity();
      pressure_dof_idx = flow_operation->get_dof_handler_idx_pressure();

      /*
       *    initialize the melt pool operation class
       */
      if (base_in->parameters.base.problem_name == "melt_pool")
        melt_pool_operation.initialize(scratch_data,
                                       base_in->parameters,
                                       ls_dof_idx,
                                       vel_dof_idx,
                                       flow_quad_idx,
                                       temp_dof_idx,
                                       temp_quad_idx,
                                       base_in->parameters.flow.start_time);

      setup_dof_system(base_in, false);

#ifdef MELT_POOL_DG_WITH_ADAFLO
      dynamic_cast<AdafloWrapper<dim> *>(flow_operation.get())->initialize(base_in);
#else
      AssertThrow(false, ExcNotImplemented());
#endif
      /*
       *  initialize the time stepping scheme
       */
      time_iterator.initialize(
        TimeIteratorData<double>{base_in->parameters.flow.start_time,
                                 base_in->parameters.flow.end_time,
                                 base_in->parameters.flow.time_step_size,
                                 base_in->parameters.flow.max_n_steps,
                                 false /*cfl_condition-->not supported yet*/});
      /*
       *  set initial conditions of the levelset function
       */
      AssertThrow(
        base_in->get_initial_condition("level_set"),
        ExcMessage(
          "It seems that your SimulationBase object does not contain "
          "a valid initial field function for the level set field. A shared_ptr to your initial field "
          "function, e.g., MyInitializeFunc<dim> must be specified as follows: "
          "  this->attach_initial_condition(std::make_shared<MyInitializeFunc<dim>>(), 'level_set') "));

      VectorType initial_solution;
      scratch_data->initialize_dof_vector(initial_solution, ls_dof_idx);

      dealii::VectorTools::project(scratch_data->get_mapping(),
                                   dof_handler,
                                   ls_constraints_dirichlet,
                                   scratch_data->get_quadrature(),
                                   *base_in->get_initial_condition("level_set"),
                                   initial_solution);
      initial_solution.update_ghost_values();
      /*
       *    initialize the levelset operation class
       */
      level_set_operation.initialize(scratch_data,
                                     initial_solution,
                                     flow_operation->get_velocity(),
                                     base_in,
                                     ls_dof_idx,
                                     ls_hanging_nodes_dof_idx,
                                     ls_quad_idx,
                                     reinit_dof_idx,
                                     reinit_hanging_nodes_dof_idx,
                                     curv_dof_idx,
                                     normal_dof_idx,
                                     vel_dof_idx,
                                     ls_dof_idx /* todo: ls_zero_bc_idx*/);
      /*
       * set initial condition of the melt pool class
       */
      if (base_in->parameters.base.problem_name == "melt_pool")
        melt_pool_operation.set_initial_condition(level_set_operation.level_set_as_heaviside);

      /*
       *  initialize postprocessor
       */
      post_processor =
        std::make_shared<Postprocessor<dim>>(scratch_data->get_mpi_comm(flow_dof_idx),
                                             base_in->parameters.paraview,
                                             scratch_data->get_mapping(),
                                             scratch_data->get_triangulation(flow_dof_idx));
      /*
       *  output results of initialization --> initial refinement is done afterwards (!)
       *  @todo: find a way to plot vectors on the refined mesh, which are only relevant for output
       *  and which must not be transferred to the new mesh everytime refine_mesh() is called.
       */
      output_results(0, base_in->parameters.base.problem_name == "melt_pool");
      /*
       *    Do initial refinement steps if requested
       */
      if (base_in->parameters.amr.do_amr && base_in->parameters.amr.n_initial_refinement_cycles > 0)
        for (int i = 0; i < base_in->parameters.amr.n_initial_refinement_cycles; ++i)
          {
            scratch_data->get_pcout()
              << "cycle: " << i << " n_dofs: " << dof_handler.n_dofs() << "(ls) + "
              << flow_operation->get_dof_handler_velocity().n_dofs() << "(vel) + "
              << flow_operation->get_dof_handler_pressure().n_dofs() << "(p)"
              << " T.size " << melt_pool_operation.temperature.size() << " solid.size "
              << melt_pool_operation.solid.size() << std::endl;

            refine_mesh(base_in);
          }
    }

    void
    setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in, const bool do_reinit = true)
    {
#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
      if (base_in->parameters.base.do_simplex)
        {
          dof_handler.distribute_dofs(Simplex::FE_P<dim>(base_in->parameters.base.degree));
          flow_dof_handler.distribute_dofs(
            Simplex::FE_P<dim>(base_in->parameters.flow.velocity_degree));
        }
      else
#endif
        {
          dof_handler.distribute_dofs(FE_Q<dim>(base_in->parameters.base.degree));
          flow_dof_handler.distribute_dofs(FE_Q<dim>(base_in->parameters.flow.velocity_degree));
        }

        /*
         *    initialize the flow operation class
         */
#ifdef MELT_POOL_DG_WITH_ADAFLO
      dynamic_cast<AdafloWrapper<dim> *>(flow_operation.get())->reinit_1();
#else
      AssertThrow(false, ExcNotImplemented());
#endif
      /*
       *  create partitioning
       */
      scratch_data->create_partitioning();
      /*
       *  make hanging nodes and dirichlet constraints (at the moment no time-dependent
       *  dirichlet constraints are supported)
       */
      ls_hanging_node_constraints.clear();
      ls_hanging_node_constraints.reinit(scratch_data->get_locally_relevant_dofs(ls_dof_idx));
      DoFTools::make_hanging_node_constraints(dof_handler, ls_hanging_node_constraints);
      ls_hanging_node_constraints.close();

      ls_constraints_dirichlet.clear();
      ls_constraints_dirichlet.reinit(scratch_data->get_locally_relevant_dofs(ls_dof_idx));
      if (base_in->get_bc("level_set") && !base_in->get_dirichlet_bc("level_set").empty())
        {
          for (const auto &bc : base_in->get_dirichlet_bc(
                 "level_set")) // @todo: add name of bc at a more central place
            {
              dealii::VectorTools::interpolate_boundary_values(
                scratch_data->get_mapping(),
                dof_handler,
                bc.first, //@todo: function map can be provided as argument directly
                *bc.second,
                ls_constraints_dirichlet);
            }
        }

      ls_constraints_dirichlet.merge(ls_hanging_node_constraints);
      ls_constraints_dirichlet.close();

      reinit_constraints_dirichlet.clear();
      reinit_constraints_dirichlet.reinit(scratch_data->get_locally_relevant_dofs());
      if (base_in->get_bc("reinitialization") &&
          !base_in->get_dirichlet_bc("reinitialization").empty())
        {
          for (const auto &bc : base_in->get_dirichlet_bc(
                 "reinitialization")) // @todo: add name of bc at a more central place
            {
              std::cout << "bc:: " << std::endl;
              dealii::VectorTools::interpolate_boundary_values(
                scratch_data->get_mapping(),
                dof_handler,
                bc.first, //@todo: function map can be provided as argument directly
                *bc.second,
                reinit_constraints_dirichlet);
            }
        }
      reinit_constraints_dirichlet.merge(ls_hanging_node_constraints);
      reinit_constraints_dirichlet.close();

      scratch_data->build();
      /*
       *    limit the level set interface to the touching regions of liquid/gas
       */
      if ((base_in->parameters.base.problem_name == "melt_pool") &&
          base_in->parameters.mp.set_level_set_to_zero_in_solid)

        {
          melt_pool_operation.remove_the_level_set_from_solid_regions(dof_handler,
                                                                      ls_constraints_dirichlet);
          melt_pool_operation.remove_the_level_set_from_solid_regions(dof_handler,
                                                                      reinit_constraints_dirichlet);
        }

      if (do_reinit)
        {
          level_set_operation.reinit();
          if (base_in->parameters.base.problem_name == "melt_pool")
            melt_pool_operation.reinit();
        }

#ifdef MELT_POOL_DG_WITH_ADAFLO
      dynamic_cast<AdafloWrapper<dim> *>(flow_operation.get())->reinit_2();
#else
      AssertThrow(false, ExcNotImplemented());
#endif
      /*
       *    initialize the force vector for calculating surface tension
       */
      scratch_data->initialize_dof_vector(force_rhs, vel_dof_idx);
    }

    /**
     * Update material parameter of the phases.
     *
     * @todo Find a better place.
     */
    void
    update_phases(const VectorType &src, const Parameters<double> &parameters) const
    {
      double dummy;

      scratch_data->get_matrix_free().template cell_loop<double, VectorType>(
        [&](const auto &matrix_free, auto &, const auto &src, auto macro_cells) {
          FECellIntegrator<dim, 1, double> ls_values(matrix_free, ls_dof_idx, flow_quad_idx);

          for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
            {
              ls_values.reinit(cell);
              ls_values.read_dof_values_plain(src);
              ls_values.evaluate(true, false);

              for (unsigned int q = 0; q < ls_values.n_q_points; ++q)
                {
                  // convert level-set value to heaviside function
                  const auto indicator = UtilityFunctions::heaviside(ls_values.get_value(q), 0.5);
                  // set density
                  flow_operation->get_density(cell, q) =
                    parameters.flow.density + parameters.flow.density_difference * indicator;

                  // set viscosity
                  flow_operation->get_viscosity(cell, q) =
                    parameters.flow.viscosity + parameters.flow.viscosity_difference * indicator;

                  // check if no spurious densities or viscosities are computed
                  for (auto dens : flow_operation->get_density(cell, q))
                    if (!((dens == parameters.flow.density) ||
                          (dens == parameters.flow.density_difference + parameters.flow.density)))
                      std::cout << "WARNING: density does not comply with input:" << dens
                                << std::endl;
                  for (auto visc : flow_operation->get_viscosity(cell, q))
                    if (!((visc == parameters.flow.viscosity) ||
                          (visc ==
                           parameters.flow.viscosity_difference + parameters.flow.viscosity)))
                      std::cout << "WARNING: viscosity does not comply with input:" << visc
                                << std::endl;
                }
            }
        },
        dummy,
        src);
    }

    /**
     * Compute gravity force.
     *
     * @todo Find a better place.
     */
    void
    compute_gravity_force(VectorType &vec, const double gravity, const bool zero_out = true) const
    {
      scratch_data->get_matrix_free().template cell_loop<VectorType, std::nullptr_t>(
        [&](const auto &matrix_free, auto &vec, const auto &, auto macro_cells) {
          FECellIntegrator<dim, dim, double> force_values(matrix_free, vel_dof_idx, flow_quad_idx);

          for (unsigned int cell = macro_cells.first; cell < macro_cells.second; ++cell)
            {
              force_values.reinit(cell);

              for (unsigned int q = 0; q < force_values.n_q_points; ++q)
                {
                  Tensor<1, dim, VectorizedArray<double>> force;

                  force[dim - 1] -= gravity * flow_operation->get_density(cell, q);
                  force_values.submit_value(force, q);
                }
              force_values.integrate_scatter(true, false, vec);
            }
        },
        vec,
        nullptr,
        zero_out);
    }

    /*
     *  This function is to create paraview output
     */
    void
    output_results(const unsigned int n_time_step, const bool do_melt_pool)
    {
      /**
       * collect all relevant output data
       */
      const auto attach_output_vectors = [&](DataOut<dim> &data_out) {
        level_set_operation.attach_output_vectors(data_out);

        if (do_melt_pool)
          melt_pool_operation.attach_output_vectors(data_out);

        flow_operation->attach_output_vectors(data_out);
      };
      /**
       * do the output operation
       */
      post_processor->process(n_time_step, attach_output_vectors);
    }
    /*
     *  perform mesh refinement
     */
    void
    refine_mesh(std::shared_ptr<SimulationBase<dim>> base_in)
    {
      const auto mark_cells_for_refinement =
        [&](parallel::distributed::Triangulation<dim> &tria) -> bool {
        Vector<float> estimated_error_per_cell(base_in->triangulation->n_active_cells());

        VectorType locally_relevant_solution;
        locally_relevant_solution.reinit(scratch_data->get_partitioner(ls_dof_idx));

        locally_relevant_solution.copy_locally_owned_data_from(level_set_operation.get_level_set());
        ls_constraints_dirichlet.distribute(locally_relevant_solution);
        locally_relevant_solution.update_ghost_values();

        for (unsigned int i = 0; i < locally_relevant_solution.local_size(); ++i)
          locally_relevant_solution.local_element(i) =
            (1.0 - locally_relevant_solution.local_element(i) *
                     locally_relevant_solution.local_element(i));

        locally_relevant_solution.update_ghost_values();

        dealii::VectorTools::integrate_difference(scratch_data->get_dof_handler(ls_dof_idx),
                                                  locally_relevant_solution,
                                                  Functions::ZeroFunction<dim>(),
                                                  estimated_error_per_cell,
                                                  scratch_data->get_quadrature(ls_quad_idx),
                                                  dealii::VectorTools::L2_norm);

        parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
          tria,
          estimated_error_per_cell,
          base_in->parameters.amr.upper_perc_to_refine,
          base_in->parameters.amr.lower_perc_to_coarsen);

        return true;
      };

      std::vector<
        std::pair<const DoFHandler<dim> *, std::function<void(std::vector<VectorType *> &)>>>
        data;

      if (base_in->parameters.base.problem_name == "melt_pool")
        data.emplace_back(&dof_handler, [&](std::vector<VectorType *> &vectors) {
          melt_pool_operation.attach_vectors(vectors); // temperature + solid
        });
      data.emplace_back(&dof_handler, [&](std::vector<VectorType *> &vectors) {
        level_set_operation.attach_vectors(vectors); // ls
      });
      data.emplace_back(&flow_operation->get_dof_handler_velocity(),
                        [&](std::vector<VectorType *> &vectors) {
                          flow_operation->attach_vectors_u(vectors);
                        });
      data.emplace_back(&flow_operation->get_dof_handler_pressure(),
                        [&](std::vector<VectorType *> &vectors) {
                          flow_operation->attach_vectors_p(vectors);
                        });

      const auto post = [&]() {
        ls_constraints_dirichlet.distribute(level_set_operation.get_level_set());
        scratch_data->get_constraint(vel_dof_idx).distribute(flow_operation->get_velocity());
        scratch_data->get_constraint(pressure_dof_idx).distribute(flow_operation->get_pressure());

        if (base_in->parameters.base.problem_name == "melt_pool")
          {
            scratch_data->get_constraint(temp_dof_idx).distribute(melt_pool_operation.temperature);
            scratch_data->get_constraint(temp_dof_idx).distribute(melt_pool_operation.solid);
          }
      };

      const auto setup_dof_system = [&]() { this->setup_dof_system(base_in); };

      refine_grid<dim, VectorType>(
        mark_cells_for_refinement, data, post, setup_dof_system, base_in->parameters.amr);
    }

    TimeIterator<double> time_iterator;
    DoFHandler<dim>      dof_handler;
    DoFHandler<dim>      flow_dof_handler;

    AffineConstraints<double> ls_constraints_dirichlet;
    AffineConstraints<double> ls_hanging_node_constraints;
    AffineConstraints<double> reinit_constraints_dirichlet;
    AffineConstraints<double> flow_dummy_constraint;

    VectorType   force_rhs;
    unsigned int ls_dof_idx;
    unsigned int ls_hanging_nodes_dof_idx;
    unsigned int ls_quad_idx;
    unsigned int reinit_dof_idx;

    const unsigned int &reinit_hanging_nodes_dof_idx = ls_hanging_nodes_dof_idx;
    const unsigned int &curv_dof_idx                 = ls_hanging_nodes_dof_idx;
    const unsigned int &normal_dof_idx               = ls_hanging_nodes_dof_idx;
    const unsigned int &temp_dof_idx                 = ls_hanging_nodes_dof_idx;
    const unsigned int &temp_quad_idx                = ls_quad_idx;

    unsigned int vel_dof_idx;
    unsigned int pressure_dof_idx;
    unsigned int flow_dof_idx;
    unsigned int flow_quad_idx;

    std::shared_ptr<ScratchData<dim>> scratch_data;
    std::shared_ptr<FlowBase<dim>>    flow_operation;
    LevelSet::LevelSetOperation<dim>  level_set_operation;
    MeltPool::MeltPoolOperation<dim>  melt_pool_operation;

    std::shared_ptr<Postprocessor<dim>> post_processor;
  };
} // namespace MeltPoolDG::Flow
