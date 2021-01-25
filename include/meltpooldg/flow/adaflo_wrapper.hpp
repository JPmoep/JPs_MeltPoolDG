/* ---------------------------------------------------------------------
 *
 * Author: Peter Münch, Magdalena Schreter, TUM, October 2020
 *
 * ---------------------------------------------------------------------*/
#pragma once

#ifdef MELT_POOL_DG_WITH_ADAFLO

#  include <deal.II/lac/generic_linear_algebra.h>

#  include <meltpooldg/flow/adaflo_wrapper_parameters.hpp>
#  include <meltpooldg/flow/flow_base.hpp>
#  include <meltpooldg/interface/scratch_data.hpp>
#  include <meltpooldg/utilities/utilityfunctions.hpp>
#  include <meltpooldg/utilities/vector_tools.hpp>

#  include <adaflo/navier_stokes.h>
#  include <adaflo/parameters.h>

namespace MeltPoolDG::Flow
{
  template <int dim>
  class AdafloWrapper : public FlowBase<dim>
  {
  private:
    using VectorType = LinearAlgebra::distributed::Vector<double>;

  public:
    /**
     * Constructor.
     */
    AdafloWrapper(ScratchData<dim, dim, double, VectorizedArray<double>> &scratch_data,
                  std::shared_ptr<SimulationBase<dim>>                    base_in)
      : scratch_data(scratch_data)
      , timer(std::cout, TimerOutput::never, TimerOutput::wall_times)
      , navier_stokes(base_in->parameters.adaflo_params.get_parameters(),
                      *const_cast<Triangulation<dim> *>(&scratch_data.get_triangulation()),
                      &timer)
    {
      /*
       * Boundary conditions for the velocity field
       */
      for (const auto &symmetry_id : base_in->get_symmetry_id("navier_stokes_u"))
        navier_stokes.set_symmetry_boundary(symmetry_id);
      for (const auto &no_slip_id : base_in->get_no_slip_id("navier_stokes_u"))
        navier_stokes.set_no_slip_boundary(no_slip_id);
      for (const auto &dirichlet_bc : base_in->get_dirichlet_bc("navier_stokes_u"))
        navier_stokes.set_velocity_dirichlet_boundary(dirichlet_bc.first, dirichlet_bc.second);
      /*
       * Boundary conditions for the pressure field
       */
      for (const auto &neumann_bc : base_in->get_neumann_bc("navier_stokes_p"))
        navier_stokes.set_open_boundary_with_normal_flux(neumann_bc.first, neumann_bc.second);
      for (const auto &fix_pressure_constant_id :
           base_in->get_fix_pressure_constant_id("navier_stokes_p"))
        navier_stokes.fix_pressure_constant(fix_pressure_constant_id);
      /*
       * Initial conditions of the navier stokes problem
       */
      AssertThrow(
        base_in->get_initial_condition("navier_stokes_u"),
        ExcMessage(
          "It seems that your SimulationBase object does not contain "
          "a valid initial field function for the level set field. A shared_ptr to your initial field "
          "function, e.g., MyInitializeFunc<dim> must be specified as follows: "
          "  this->attach_initial_condition(std::make_shared<MyInitializeFunc<dim>>(), 'navier_stokes_u') "));

      this->dof_index_u = scratch_data.attach_dof_handler(navier_stokes.get_dof_handler_u());
      this->dof_index_p = scratch_data.attach_dof_handler(navier_stokes.get_dof_handler_p());

      scratch_data.attach_constraint_matrix(navier_stokes.get_constraints_u());
      scratch_data.attach_constraint_matrix(navier_stokes.get_constraints_p());

      const auto &adaflo_params = base_in->parameters.adaflo_params.get_parameters();

      this->quad_index_u =
        adaflo_params.use_simplex_mesh ?
          scratch_data.attach_quadrature(Simplex::QGauss<dim>(adaflo_params.velocity_degree + 1)) :
          scratch_data.attach_quadrature(QGauss<dim>(adaflo_params.velocity_degree + 1));
      this->quad_index_p =
        adaflo_params.use_simplex_mesh ?
          scratch_data.attach_quadrature(Simplex::QGauss<dim>(adaflo_params.velocity_degree)) :
          scratch_data.attach_quadrature(QGauss<dim>(adaflo_params.velocity_degree));
    }

    void
    initialize(std::shared_ptr<SimulationBase<dim>> base_in)
    {
      dealii::VectorTools::interpolate(navier_stokes.mapping,
                                       navier_stokes.get_dof_handler_u(),
                                       *base_in->get_initial_condition("navier_stokes_u"),
                                       navier_stokes.solution.block(0));

      navier_stokes.get_constraints_u().distribute(navier_stokes.solution.block(0)); // TODO
      // needed?
      navier_stokes.solution.update_ghost_values();
      navier_stokes.solution_old.update_ghost_values();
    }

    void
    reinit_1()
    {
      // clear constraints and setup hanging node constraints
      navier_stokes.distribute_dofs();
      // fill constraints_u and constraints_p
      navier_stokes.initialize_data_structures();
    }

    void
    reinit_2()
    {
      navier_stokes.initialize_matrix_free(
        &scratch_data.get_matrix_free(), dof_index_u, dof_index_p, quad_index_u, quad_index_p);
    }

    /**
     * Solver time step
     */
    void
    solve() override
    {
      navier_stokes.get_constraints_u().set_zero(navier_stokes.user_rhs.block(0));
      navier_stokes.advance_time_step();
    }

    const LinearAlgebra::distributed::Vector<double> &
    get_velocity() const override
    {
      return navier_stokes.solution.block(0);
    }

    LinearAlgebra::distributed::Vector<double> &
    get_velocity() override
    {
      return navier_stokes.solution.block(0);
    }

    const DoFHandler<dim> &
    get_dof_handler_velocity() const override
    {
      return navier_stokes.get_dof_handler_u();
    }

    const unsigned int &
    get_dof_handler_idx_velocity() const override
    {
      return dof_index_u;
    }

    const unsigned int &
    get_quad_idx_velocity() const override
    {
      return quad_index_u;
    }

    const AffineConstraints<double> &
    get_constraints_velocity() const override
    {
      return navier_stokes.get_constraints_u();
    }

    AffineConstraints<double> &
    get_constraints_velocity() override
    {
      return navier_stokes.modify_constraints_u();
    }

    const LinearAlgebra::distributed::Vector<double> &
    get_pressure() const override
    {
      return navier_stokes.solution.block(1);
    }

    LinearAlgebra::distributed::Vector<double> &
    get_pressure() override
    {
      return navier_stokes.solution.block(1);
    }

    const DoFHandler<dim> &
    get_dof_handler_pressure() const override
    {
      return navier_stokes.get_dof_handler_p();
    }

    const unsigned int &
    get_dof_handler_idx_pressure() const override
    {
      return dof_index_p;
    }

    const AffineConstraints<double> &
    get_constraints_pressure() const override
    {
      return navier_stokes.get_constraints_p();
    }

    AffineConstraints<double> &
    get_constraints_pressure() override
    {
      return navier_stokes.modify_constraints_p();
    }

    void
    set_force_rhs(const LinearAlgebra::distributed::Vector<double> &vec) override
    {
      navier_stokes.user_rhs.block(0) = vec;
    }

    VectorizedArray<double> &
    get_density(const unsigned int cell, const unsigned int q) override
    {
      return navier_stokes.get_matrix().begin_densities(cell)[q];
    }

    const VectorizedArray<double> &
    get_density(const unsigned int cell, const unsigned int q) const override
    {
      return navier_stokes.get_matrix().begin_densities(cell)[q];
    }

    VectorizedArray<double> &
    get_viscosity(const unsigned int cell, const unsigned int q) override
    {
      return navier_stokes.get_matrix().begin_viscosities(cell)[q];
    }

    const VectorizedArray<double> &
    get_viscosity(const unsigned int cell, const unsigned int q) const override
    {
      return navier_stokes.get_matrix().begin_viscosities(cell)[q];
    }

    void
    attach_vectors_u(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) override
    {
      navier_stokes.solution.block(0).update_ghost_values();
      navier_stokes.solution_old.block(0).update_ghost_values();
      vectors.push_back(&navier_stokes.solution.block(0));
      vectors.push_back(&navier_stokes.solution_old.block(0));
    }

    void
    attach_vectors_p(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) override
    {
      navier_stokes.solution.block(1).update_ghost_values();
      navier_stokes.solution_old.block(1).update_ghost_values();
      vectors.push_back(&navier_stokes.solution.block(1));
      vectors.push_back(&navier_stokes.solution_old.block(1));
    }

    void
    attach_output_vectors(DataOut<dim> &data_out)
    {
      MeltPoolDG::VectorTools::update_ghost_values(get_velocity(),
                                                   get_pressure(),
                                                   navier_stokes.user_rhs.block(0));

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        vector_component_interpretation(dim,
                                        DataComponentInterpretation::component_is_part_of_vector);
      /**
       *  velocity
       */
      data_out.add_data_vector(get_dof_handler_velocity(),
                               get_velocity(),
                               std::vector<std::string>(dim, "velocity"),
                               vector_component_interpretation);
      /**
       *  pressure
       */
      data_out.add_data_vector(get_dof_handler_pressure(), get_pressure(), "pressure");
      /**
       *  force rhs
       */
      data_out.add_data_vector(get_dof_handler_velocity(),
                               navier_stokes.user_rhs.block(0),
                               std::vector<std::string>(dim, "force_rhs_velocity"),
                               vector_component_interpretation);
      /**
       *  density
       */
      scratch_data.initialize_dof_vector(density, dof_index_p);
      UtilityFunctions::fill_dof_vector_from_cell_operation<dim, 1>(
        density,
        scratch_data.get_matrix_free(),
        dof_index_p,
        quad_index_p,
        1, // fe_degree,
        2, /// n_q_points_1D,
        [&](const unsigned int cell, const unsigned int quad) -> const VectorizedArray<double> & {
          return get_density(cell, quad);
        });
      density.update_ghost_values();
      /**
       *  viscosity
       */
      scratch_data.initialize_dof_vector(viscosity, dof_index_p);
      data_out.add_data_vector(get_dof_handler_pressure(), density, "density");
      UtilityFunctions::fill_dof_vector_from_cell_operation<dim, 1>(
        viscosity,
        scratch_data.get_matrix_free(),
        dof_index_p,
        quad_index_p,
        1, // fe_degree,
        2, /// n_q_points_1D,
        [&](const unsigned int cell, const unsigned int quad) -> const VectorizedArray<double> & {
          return get_viscosity(cell, quad);
        });
      viscosity.update_ghost_values();
      data_out.add_data_vector(get_dof_handler_pressure(), viscosity, "viscosity");
      /**
       *  viscosity
       */
    }

  private:
    ScratchData<dim, dim, double, VectorizedArray<double>> &scratch_data;
    /**
     * Timer
     */
    TimerOutput timer;

    /**
     * Reference to the actual Navier-Stokes solver from adaflo
     */
    NavierStokes<dim> navier_stokes;

    unsigned int dof_index_u;
    unsigned int dof_index_p;

    unsigned int quad_index_u;
    unsigned int quad_index_p;
    /**
     * density and viscosity mainly for output purposes
     */
    VectorType density;
    VectorType viscosity;
  };

  /**
   * Dummy specialization for 1. Needed to be able to compile
   * since the adaflo Navier-Stokes solver is not compiled for
   * 1D - due to the dependcy to parallel::distributed::Triangulation
   * and p4est.
   */
  template <>
  class AdafloWrapper<1> : public FlowBase<1>
  {
  public:
    /**
     * Dummy constructor.
     */
    AdafloWrapper(ScratchData<1, 1, double, VectorizedArray<double>> &scratch_data,
                  std::shared_ptr<SimulationBase<1>>                  base_in)
    {
      (void)scratch_data;
      (void)base_in;

      AssertThrow(false, ExcNotImplemented());
    }

    void
    initialize(std::shared_ptr<SimulationBase<1>> base_in)
    {
      (void)base_in;
      AssertThrow(false, ExcNotImplemented());
    }

    void
    reinit_1()
    {
      AssertThrow(false, ExcNotImplemented());
    }

    void
    reinit_2()
    {
      AssertThrow(false, ExcNotImplemented());
    }

    const LinearAlgebra::distributed::Vector<double> &
    get_velocity() const override
    {
      AssertThrow(false, ExcNotImplemented());
    }

    LinearAlgebra::distributed::Vector<double> &
    get_velocity() override
    {
      AssertThrow(false, ExcNotImplemented());
    }

    const DoFHandler<1> &
    get_dof_handler_velocity() const override
    {
      AssertThrow(false, ExcNotImplemented());
    }

    const unsigned int &
    get_dof_handler_idx_velocity() const override
    {
      AssertThrow(false, ExcNotImplemented());
    }

    const unsigned int &
    get_quad_idx_velocity() const override
    {
      AssertThrow(false, ExcNotImplemented());
    }

    const AffineConstraints<double> &
    get_constraints_velocity() const override
    {
      AssertThrow(false, ExcNotImplemented());
    }

    AffineConstraints<double> &
    get_constraints_velocity() override
    {
      AssertThrow(false, ExcNotImplemented());
    }

    const LinearAlgebra::distributed::Vector<double> &
    get_pressure() const override
    {
      AssertThrow(false, ExcNotImplemented());
    }

    LinearAlgebra::distributed::Vector<double> &
    get_pressure() override
    {
      AssertThrow(false, ExcNotImplemented());
    }

    const DoFHandler<1> &
    get_dof_handler_pressure() const override
    {
      AssertThrow(false, ExcNotImplemented());
    }

    const unsigned int &
    get_dof_handler_idx_pressure() const override
    {
      AssertThrow(false, ExcNotImplemented());
    }

    const AffineConstraints<double> &
    get_constraints_pressure() const override
    {
      AssertThrow(false, ExcNotImplemented());
    }

    AffineConstraints<double> &
    get_constraints_pressure() override
    {
      AssertThrow(false, ExcNotImplemented());
    }

    void
    set_force_rhs(const LinearAlgebra::distributed::Vector<double> &) override
    {
      AssertThrow(false, ExcNotImplemented());
    }

    void
    solve() override
    {
      AssertThrow(false, ExcNotImplemented());
    }

    VectorizedArray<double> &
    get_density(const unsigned int cell, const unsigned int q) override
    {
      AssertThrow(false, ExcNotImplemented());
      (void)cell;
      (void)q;
      return dummy;
    }

    const VectorizedArray<double> &
    get_density(const unsigned int cell, const unsigned int q) const override
    {
      AssertThrow(false, ExcNotImplemented());
      (void)cell;
      (void)q;
      return dummy;
    }

    VectorizedArray<double> &
    get_viscosity(const unsigned int cell, const unsigned int q) override
    {
      AssertThrow(false, ExcNotImplemented());
      (void)cell;
      (void)q;
      return dummy;
    }

    const VectorizedArray<double> &
    get_viscosity(const unsigned int cell, const unsigned int q) const override
    {
      AssertThrow(false, ExcNotImplemented());
      (void)cell;
      (void)q;
      return dummy;
    }

    void
    attach_vectors_u(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) override
    {
      Assert(false, ExcNotImplemented());
      (void)vectors;
    }

    void
    attach_vectors_p(std::vector<LinearAlgebra::distributed::Vector<double> *> &vectors) override
    {
      Assert(false, ExcNotImplemented());
      (void)vectors;
    }

    void
    attach_output_vectors(DataOut<1> &data_out)
    {
      Assert(false, ExcNotImplemented());
      (void)data_out;
    }

  private:
    VectorizedArray<double> dummy;
  };
} // namespace MeltPoolDG::Flow

#endif
