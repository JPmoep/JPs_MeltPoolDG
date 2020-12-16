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

#include <deal.II/fe/fe_dgq.h>
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

namespace MeltPoolDG
{
  namespace Flow
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

        output_results(0, base_in->parameters);

        while (!time_iterator.is_finished())
          {
            const auto dt = time_iterator.get_next_time_increment();
            const auto n  = time_iterator.get_current_time_step_number();

            scratch_data->get_pcout()
              << "t= " << std::setw(10) << std::left << time_iterator.get_current_time();

            // ... solve level-set problem with the given advection field
            level_set_operation.solve(dt, advection_velocity);

            // update
            update_phases(level_set_operation.level_set_as_heaviside, base_in->parameters);

            // accumulate forces: a) gravity force
            compute_gravity_force(force_rhs, base_in->parameters.base.gravity);

            // ... b) surface tension
            if (base_in->parameters.flow.temperature_dependent_surface_tension_coefficient == 0)
              level_set_operation.compute_surface_tension(
                force_rhs,
                base_in->parameters.flow.surface_tension_coefficient,
                flow_dof_idx,
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

                // ... d) temperature-dependent surface tension
                if (base_in->parameters.flow.temperature_dependent_surface_tension_coefficient >
                    0.0)
                  melt_pool_operation.compute_temperature_dependent_surface_tension(
                    force_rhs,
                    level_set_operation.level_set_as_heaviside,
                    level_set_operation.get_curvature(),
                    base_in->parameters.flow.surface_tension_coefficient,
                    base_in->parameters.flow.temperature_dependent_surface_tension_coefficient,
                    base_in->parameters.flow.surface_tension_reference_temperature,
                    ls_dof_idx,
                    flow_dof_idx,
                    flow_quad_idx,
                    dof_no_bc_idx, /*temp_dof_idx*/
                    false /*false means add to force vector*/);
              }

            //  ... and set forces within the Navier-Stokes solver
            flow_operation->set_force_rhs(force_rhs);

            // solver Navier-Stokes problem
            flow_operation->solve();

            // extract velocity ...
            flow_operation->get_velocity(advection_velocity);

            // ... and output the results to vtk files.
            output_results(n, base_in->parameters);

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
          scratch_data->attach_dof_handler(flow_dof_handler);

          dof_no_bc_idx = scratch_data->attach_constraint_matrix(ls_hanging_node_constraints);
          ls_dof_idx    = scratch_data->attach_constraint_matrix(ls_constraints_dirichlet);
          flow_dof_idx  = scratch_data->attach_constraint_matrix(flow_dummy_constraint);

          /*
           *  create quadrature rule
           */
#ifdef DEAL_II_WITH_SIMPLEX_SUPPORT
          if (base_in->parameters.base.do_simplex)
            {
              ls_quad_idx = scratch_data->attach_quadrature(
                Simplex::QGauss<1>(base_in->parameters.base.n_q_points_1d));
              flow_quad_idx = scratch_data->attach_quadrature(
                Simplex::QGauss<1>(base_in->parameters.flow.velocity_degree + 1));
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
        flow_operation = std::make_shared<AdafloWrapper<dim>>(*scratch_data, flow_dof_idx, base_in);
#else
        AssertThrow(false, ExcNotImplemented());
#endif

        setup_dof_system(base_in);

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
                                       advection_velocity,
                                       base_in->parameters,
                                       ls_dof_idx,
                                       dof_no_bc_idx,
                                       ls_quad_idx,
                                       flow_dof_idx,
                                       base_in);
        /*
         *    initialize the melt pool operation class
         */
        if (base_in->parameters.base.problem_name == "melt_pool")
          {
            melt_pool_operation.initialize(scratch_data,
                                           base_in->parameters,
                                           ls_dof_idx,
                                           flow_dof_idx,
                                           flow_quad_idx,
                                           dof_no_bc_idx, /*temp_dof_idx @todo: may be changed as
                                                             soon as heat problem is introduced*/
                                           ls_quad_idx, /*temp_quad_idx@todo: may be changed as soon
                                                           as heat problem is introduced*/
                                           level_set_operation.level_set_as_heaviside);
          }
      }

      void
      setup_dof_system(std::shared_ptr<SimulationBase<dim>> base_in)
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
        ls_hanging_node_constraints.reinit(scratch_data->get_locally_relevant_dofs());
        DoFTools::make_hanging_node_constraints(dof_handler, ls_hanging_node_constraints);
        ls_hanging_node_constraints.close();

        ls_constraints_dirichlet.clear();
        ls_constraints_dirichlet.reinit(scratch_data->get_locally_relevant_dofs());
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

        scratch_data->build();

#ifdef MELT_POOL_DG_WITH_ADAFLO
        dynamic_cast<AdafloWrapper<dim> *>(flow_operation.get())->reinit_2();
#else
        AssertThrow(false, ExcNotImplemented());
#endif
        /*
         *  initialize the velocity dof vector
         */
        scratch_data->initialize_dof_vector(advection_velocity, flow_dof_idx);

        if (base_in->parameters.base.problem_name == "melt_pool")
          {
            /*
             *    set the fluid velocity and the pressure in solid regions to zero
             */
            melt_pool_operation.set_flow_field_in_solid_regions_to_zero(
              flow_operation->get_dof_handler_velocity(),
              flow_operation->get_constraints_velocity());
            melt_pool_operation.set_flow_field_in_solid_regions_to_zero(
              flow_operation->get_dof_handler_pressure(),
              flow_operation->get_constraints_pressure());
          }
        /*
         *    initialize the force vector for calculating surface tension
         */
        scratch_data->initialize_dof_vector(force_rhs, flow_dof_idx);
        /*
         *    initialize the density and viscosity (for postprocessing)
         */
        scratch_data->initialize_dof_vector(density, flow_dof_idx);
        scratch_data->initialize_dof_vector(viscosity, flow_dof_idx);
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
                    const auto densities = flow_operation->get_density(cell, q);
                    for (auto dens : densities)
                      if (!((dens == parameters.flow.density) ||
                            (dens == parameters.flow.density_difference + parameters.flow.density)))
                        std::cout << "density is wrong:" << dens << std::endl;
                    const auto viscosities = flow_operation->get_viscosity(cell, q);
                    for (auto visc : viscosities)
                      if (!((visc == parameters.flow.viscosity) ||
                            (visc ==
                             parameters.flow.viscosity_difference + parameters.flow.viscosity)))
                        std::cout << "viscosity is wrong:" << visc << std::endl;
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
      compute_gravity_force(BlockVectorType &vec,
                            const double     gravity,
                            const bool       zero_out = true) const
      {
        scratch_data->get_matrix_free().template cell_loop<BlockVectorType, std::nullptr_t>(
          [&](const auto &matrix_free, auto &vec, const auto &, auto macro_cells) {
            FECellIntegrator<dim, dim, double> force_values(matrix_free,
                                                            flow_dof_idx,
                                                            flow_quad_idx);

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
      output_results(const unsigned int time_step, const Parameters<double> &parameters)
      {
        if (parameters.paraview.do_output && !(time_step % parameters.paraview.write_frequency))
          {
            fill_dof_vector_from_cell_operation(density,
                                                parameters.flow.velocity_degree,
                                                parameters.flow.velocity_n_q_points_1d,
                                                "density");
            fill_dof_vector_from_cell_operation(viscosity,
                                                parameters.flow.velocity_degree,
                                                parameters.flow.velocity_n_q_points_1d,
                                                "viscosity");

            const VectorType &pressure = flow_operation->get_pressure();

            // update ghost values
            VectorTools::update_ghost_values(advection_velocity,
                                             force_rhs,
                                             density,
                                             viscosity,
                                             pressure,
                                             level_set_operation.get_level_set(),
                                             level_set_operation.get_curvature(),
                                             level_set_operation.get_normal_vector(),
                                             level_set_operation.level_set_as_heaviside,
                                             level_set_operation.distance_to_level_set);

            if (parameters.base.problem_name == "melt_pool")
              VectorTools::update_ghost_values(melt_pool_operation.temperature,
                                               melt_pool_operation.solid);
            /*
             *  output advected field
             */
            DataOut<dim> data_out;

            DataOutBase::VtkFlags flags;
            if (parameters.base.do_simplex == false)
              flags.write_higher_order_cells = true;
            data_out.set_flags(flags);

            data_out.attach_dof_handler(dof_handler);
            /*
             * level set
             */
            data_out.add_data_vector(level_set_operation.get_level_set(), "level_set");
            /*
             * curvature
             */
            data_out.add_data_vector(level_set_operation.get_curvature(), "curvature");
            /*
             *  normal vector field
             */
            for (unsigned int d = 0; d < dim; ++d)
              data_out.add_data_vector(level_set_operation.get_normal_vector().block(d),
                                       "normal_" + std::to_string(d));
            /*
             *  flow velocity
             */
            for (auto d = 0; d < dim; ++d)
              data_out.add_data_vector(flow_dof_handler,
                                       advection_velocity.block(d),
                                       "advection_velocity_" + std::to_string(d));

            /*
             * force vector (surface tension + gravity force)
             */
            for (auto d = 0; d < dim; ++d)
              data_out.add_data_vector(flow_dof_handler,
                                       force_rhs.block(d),
                                       "force_" + std::to_string(d));
            /*
             * density
             */
            data_out.add_data_vector(flow_dof_handler, density, "density");
            /*
             * viscosity
             */
            data_out.add_data_vector(flow_dof_handler, viscosity, "viscosity");
            /*
             * heaviside
             */
            data_out.add_data_vector(dof_handler,
                                     level_set_operation.level_set_as_heaviside,
                                     "heaviside");
            /*
             * distance to zero level set
             */
            data_out.add_data_vector(dof_handler,
                                     level_set_operation.distance_to_level_set,
                                     "distance");
            /*
             * pressure
             */
            data_out.add_data_vector(flow_operation->get_dof_handler_pressure(),
                                     pressure,
                                     "pressure");
            /*
             * temperature
             */
            if (parameters.base.problem_name == "melt_pool")
              {
                data_out.add_data_vector(dof_handler,
                                         melt_pool_operation.temperature,
                                         "temperature");
                /*
                 * solid
                 */
                data_out.add_data_vector(dof_handler, melt_pool_operation.solid, "solid");
              }

            data_out.build_patches(scratch_data->get_mapping());
            data_out.write_vtu_with_pvtu_record("./",
                                                parameters.paraview.filename,
                                                time_step / parameters.paraview.write_frequency,
                                                scratch_data->get_mpi_comm(),
                                                parameters.paraview.n_digits_timestep,
                                                parameters.paraview.n_groups);

            // clear ghost values
            VectorTools::zero_out_ghosts(advection_velocity,
                                         force_rhs,
                                         density,
                                         viscosity,
                                         pressure,
                                         level_set_operation.get_level_set(),
                                         level_set_operation.get_curvature(),
                                         level_set_operation.get_normal_vector(),
                                         level_set_operation.level_set_as_heaviside,
                                         level_set_operation.distance_to_level_set);

            if (parameters.base.problem_name == "melt_pool")
              VectorTools::zero_out_ghosts(melt_pool_operation.temperature,
                                           melt_pool_operation.solid);
          }
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
          locally_relevant_solution.reinit(scratch_data->get_partitioner());

          Assert(false, ExcNotImplemented());

          // TODO
          // locally_relevant_solution.copy_locally_owned_data_from(
          //   advec_diff_operation->get_advected_field());
          // constraints.distribute(locally_relevant_solution);
          locally_relevant_solution.update_ghost_values();

          KellyErrorEstimator<dim>::estimate(scratch_data->get_dof_handler(),
                                             scratch_data->get_face_quadrature(),
                                             {},
                                             locally_relevant_solution,
                                             estimated_error_per_cell);

          parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
            tria,
            estimated_error_per_cell,
            base_in->parameters.amr.upper_perc_to_refine,
            base_in->parameters.amr.lower_perc_to_coarsen);

          return true;
        };

        const auto attach_vectors = [&](std::vector<VectorType *> &vectors) {
          flow_operation->attach_vectors(vectors);
        };

        const auto post = [&]() { Assert(false, ExcNotImplemented()); };

        const auto setup_dof_system = [&]() { this->setup_dof_system(base_in); };

        refine_grid<dim, VectorType>(mark_cells_for_refinement,
                                     attach_vectors,
                                     post,
                                     setup_dof_system,
                                     base_in->parameters.amr,
                                     dof_handler);
      }

      //@todo this function might be designed more generic and shifted to vector tools
      void
      fill_dof_vector_from_cell_operation(VectorType & vec,
                                          unsigned int fe_degree,
                                          unsigned int n_q_points_1D,
                                          std::string  cell_operation = "density") const
      {
        FE_DGQArbitraryNodes<1> fe_coarse(QGauss<1>(n_q_points_1D).get_points());
        FE_Q<1>                 fe_fine(fe_degree);

        /// create 1D projection matrix for sum factorization
        FullMatrix<double> matrix(fe_fine.dofs_per_cell, fe_coarse.dofs_per_cell);
        FETools::get_projection_matrix(fe_coarse, fe_fine, matrix);

        AlignedVector<VectorizedArray<double>> projection_matrix_1d(fe_fine.dofs_per_cell *
                                                                    fe_coarse.dofs_per_cell);

        for (unsigned int i = 0, k = 0; i < fe_coarse.dofs_per_cell; ++i)
          for (unsigned int j = 0; j < fe_fine.dofs_per_cell; ++j, ++k)
            projection_matrix_1d[k] = matrix(j, i);


        FECellIntegrator<dim, 1, double> fe_eval(scratch_data->get_matrix_free(),
                                                 flow_dof_idx,
                                                 flow_quad_idx);

        for (unsigned int cell = 0; cell < scratch_data->get_matrix_free().n_cell_batches(); ++cell)
          {
            fe_eval.reinit(cell);

            for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
              {
                if (cell_operation == "density") //@todo: replace by functor argument
                  fe_eval.begin_values()[q] = flow_operation->get_density(cell, q);
                else if (cell_operation == "viscosity")
                  fe_eval.begin_values()[q] = flow_operation->get_viscosity(cell, q);
                else
                  AssertThrow(
                    false,
                    ExcMessage(
                      "The requested variable for fill_dof_vector_from_cell_operation is not supported."));
              }
            // perform basis change from quadrature points to support points
            internal::FEEvaluationImplBasisChange<
              internal::evaluate_general,
              internal::EvaluatorQuantity::value,
              dim,
              0,
              0,
              VectorizedArray<double>,
              VectorizedArray<double>>::do_forward(1 /*n_components*/,
                                                   projection_matrix_1d,
                                                   fe_eval.begin_values(),
                                                   fe_eval.begin_dof_values(),
                                                   n_q_points_1D, // number of quadrature points
                                                   fe_degree + 1  // number of support points
            );

            // write values back into global vector
            fe_eval.set_dof_values(vec);
          }

        vec.compress(VectorOperation::max);
      }

      TimeIterator<double> time_iterator;
      DoFHandler<dim>      dof_handler;
      DoFHandler<dim>      flow_dof_handler;

      AffineConstraints<double> ls_constraints_dirichlet;
      AffineConstraints<double> ls_hanging_node_constraints;
      AffineConstraints<double> flow_dummy_constraint;

      BlockVectorType advection_velocity;
      BlockVectorType force_rhs;
      VectorType      density;
      VectorType      viscosity;

      unsigned int ls_dof_idx;
      unsigned int dof_no_bc_idx;
      unsigned int flow_dof_idx;
      unsigned int ls_quad_idx;
      unsigned int flow_quad_idx;

      std::shared_ptr<ScratchData<dim>> scratch_data;
      std::shared_ptr<FlowBase<dim>>    flow_operation;
      LevelSet::LevelSetOperation<dim>  level_set_operation;
      MeltPool::MeltPoolOperation<dim>  melt_pool_operation;
    };
  } // namespace Flow
} // namespace MeltPoolDG
