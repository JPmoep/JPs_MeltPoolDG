#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>

#include <meltpooldg/flow/adaflo_wrapper_parameters.hpp>
// c++
#include <fstream>
#include <iostream>

namespace MeltPoolDG
{
  using namespace dealii;

  template <typename number = double>
  struct BaseData
  {
    std::string  application_name    = "none";
    std::string  problem_name        = "none";
    unsigned int dimension           = 2;
    unsigned int global_refinements  = 1;
    unsigned int degree              = 1;
    int          n_q_points_1d       = -1;
    bool         do_print_parameters = true;
    bool         do_simplex          = false;
    number       gravity             = 0.981;
  };

  template <typename number = double>
  struct AdaptiveMeshingData
  {
    bool         do_amr                    = false;
    double       upper_perc_to_refine      = 0.0;
    double       lower_perc_to_coarsen     = 0.0;
    unsigned int max_grid_refinement_level = 12;
    unsigned int min_grid_refinement_level = 1;
  };

  template <typename number = double>
  struct LevelSetData
  {
    bool   do_reinitialization    = false;
    number artificial_diffusivity = 0.0;
    number theta                  = 0.5;
    number start_time             = 0.0;
    number end_time               = 1.0;
    number time_step_size         = 0.01;
    bool   enable_CFL_condition   = false;
    bool   do_print_l2norm        = false;
    bool   do_matrix_free         = false;
  };

  template <typename number = double>
  struct ReinitializationData
  {
    unsigned int max_n_steps          = 5;
    number       constant_epsilon     = -1.0;
    number       scale_factor_epsilon = 0.5;
    number       dtau                 = -1.0;
    std::string  modeltype            = "olsson2007";
    bool         do_matrix_free       = false;
    bool         do_print_l2norm      = false;
  };

  template <typename number = double>
  struct AdvectionDiffusionData
  {
    number       diffusivity     = 0.0;
    number       theta           = 0.5;
    number       start_time      = 0.0;
    number       end_time        = 1.0;
    number       time_step_size  = 0.01;
    unsigned int max_n_steps     = 1000000;
    bool         do_matrix_free  = false;
    bool         do_print_l2norm = true;
  };

  template <typename number = double>
  struct FlowData
  {
    number      density = 0.0;
    number      density_difference = 0.0;
    number      viscosity = 0.0;
    number      viscosity_difference = 0.0;
    number      surface_tension_coefficient = 0.0;
    std::string solver_type = "incompressible";
    number      start_time      = 0.0;
    number      end_time        = 1.0;
    number      time_step_size  = 0.05;
    unsigned int max_n_steps     = 1000000;
  };

  template <typename number = double>
  struct NormalVectorData
  {
    number damping_scale_factor = 0.5;
    bool   do_matrix_free       = false;
    bool   do_print_l2norm      = true;
  };

  template <typename number = double>
  struct CurvatureData
  {
    number damping_scale_factor = 0.0;
    bool   do_matrix_free       = false;
    bool   do_print_l2norm      = true;
  };

  template <typename number = double>
  struct ParaviewData
  {
    bool        do_output           = false;
    std::string filename            = "solution";
    int         write_frequency     = 1;
    bool        do_initial_state    = true;
    bool        print_levelset      = true;
    bool        print_normal_vector = false;
    bool        print_curvature     = false;
    bool        print_advection     = false;
    bool        print_exactsolution = false;
    bool        print_boundary_id   = false;
    int         n_digits_timestep   = 4;
    int         n_groups            = 1;
  };

  template <typename number = double>
  struct OutputData
  {
    bool        do_walltime              = 0;
    bool        do_compute_error         = 0;
    bool        do_compute_volume_output = 0;
    std::string filename_volume_output   = "my_volumes.txt";
  };

  template <typename number = double>
  struct Parameters
  {
    void
    process_parameters_file(const std::string &parameter_filename)
    {
      add_parameters();

      check_for_file(parameter_filename);

      std::ifstream file;
      file.open(parameter_filename);

      if (parameter_filename.substr(parameter_filename.find_last_of(".") + 1) == "json")
        prm.parse_input_from_json(file, true);
      else if (parameter_filename.substr(parameter_filename.find_last_of(".") + 1) == "prm")
        prm.parse_input(parameter_filename);
      else
        AssertThrow(false, ExcMessage("Parameterhandler cannot handle current file ending"));
      /*
       *  set the number of quadrature points in 1d
       */
      if (base.n_q_points_1d == -1)
        base.n_q_points_1d = base.degree + 1;
      /*
       *  set the min grid refinement level if not user-specified
       */
      if (amr.min_grid_refinement_level == 1)
        amr.min_grid_refinement_level = base.global_refinements;

      /*
       *  parameters for adaflo
       */
      adaflo_params.parse_parameters(parameter_filename);

      if (base.problem_name=="two_phase_flow")
      {
      // WARNING: by setting the differences to a non-zero value we force
      //   adaflo to assume that we are running a simulation with variable
      //   coefficients, i.e., it allocates memory for the data structures
      //   variable_densities and variable_viscosities, which are accessed 
      //   during NavierStokesMatrix::begin_densities() and
      //   NavierStokesMatrix::begin_viscosity(). However, we do not actually
      //   use these values, since we fill the density and viscosity 
      //   differently.
        adaflo_params.params.density_diff         = 1.0; 
        adaflo_params.params.viscosity_diff       = 1.0; 
        adaflo_params.params.density              = flow.density;
        adaflo_params.params.viscosity            = flow.viscosity;
        adaflo_params.params.start_time           = flow.start_time;
        adaflo_params.params.end_time             = flow.end_time;
        adaflo_params.params.time_step_size_start = flow.time_step_size;
      }
    }

    void
    print_parameters(const dealii::ConditionalOStream &pcout)
    {
      if (base.do_print_parameters)
        prm.print_parameters(pcout.get_stream(), ParameterHandler::OutputStyle::Text);
    }

    void
    check_for_file(const std::string &parameter_filename) const
    {
      std::ifstream parameter_file(parameter_filename.c_str());

      if (!parameter_file)
        {
          parameter_file.close();

          std::ostringstream message;
          message << "Input parameter file <" << parameter_filename
                  << "> not found. Please make sure the file exists!" << std::endl;

          AssertThrow(false, ExcMessage(message.str().c_str()));
        }
    }

    void
    add_parameters()
    {
      /*
       *    base
       */
      prm.enter_subsection("base");
      {
        prm.add_parameter(
          "application name",
          base.application_name,
          "Sets the base name for the application that will be fed to the problem type.");
        prm.add_parameter("problem name",
                          base.problem_name,
                          "Sets the base name for the problem that should be solved.");
        prm.add_parameter("dimension", base.dimension, "Defines the dimension of the problem");
        prm.add_parameter("global refinements",
                          base.global_refinements,
                          "Defines the number of initial global refinements");
        prm.add_parameter("degree", base.degree, "Defines the interpolation degree");
        prm.add_parameter("n q points 1d",
                          base.n_q_points_1d,
                          "Defines the number of quadrature points");
        prm.add_parameter("do print parameters",
                          base.do_print_parameters,
                          "Sets this parameter to true to list parameters in output");
        prm.add_parameter("do simplex", base.do_simplex, "Use simplices");
        prm.add_parameter("gravity", base.gravity, "Set the value for the gravity");
      }
      prm.leave_subsection();
      /*
       *    adaptive meshing
       */
      prm.enter_subsection("adaptive meshing");
      {
        prm.add_parameter("do amr",
                          amr.do_amr,
                          "Sets this parameter to true to activate adaptive meshing");
        prm.add_parameter("upper perc to refine",
                          amr.upper_perc_to_refine,
                          "Defines the (upper) percentage of elements that should be refined");
        prm.add_parameter("lower perc to coarsen",
                          amr.lower_perc_to_coarsen,
                          "Defines the (lower) percentage of elements that should be coarsened");
        prm.add_parameter(
          "max grid refinement level",
          amr.max_grid_refinement_level,
          "Defines the number of maximum refinement steps one grid cell will be undergone.");
      }
      prm.leave_subsection();
      /*
       *   advection diffusion
       */
      prm.enter_subsection("advection diffusion");
      {
        prm.add_parameter("advec diff diffusivity",
                          advec_diff.diffusivity,
                          "Defines the diffusivity for the advection diffusion equation ");
        prm.add_parameter("advec diff theta",
                          advec_diff.theta,
                          "Sets the theta value for the time stepping scheme: 0=explicit euler; "
                          "1=implicit euler; 0.5=Crank-Nicholson;");
        prm.add_parameter("advec diff start time",
                          advec_diff.start_time,
                          "Defines the start time for the solution of the levelset problem");
        prm.add_parameter("advec diff end time",
                          advec_diff.end_time,
                          "Sets the end time for the solution of the advection diffusion problem");
        prm.add_parameter(
          "advec diff time step size",
          advec_diff.time_step_size,
          "Sets the step size for time stepping. For non-uniform "
          "time stepping, this parameter determines the size of the first time step.");
        prm.add_parameter("advec diff max n steps",
                          advec_diff.max_n_steps,
                          "Sets the maximum number of advection diffusion steps");
        prm.add_parameter(
          "advec diff do matrix free",
          advec_diff.do_matrix_free,
          "Set this parameter if a matrix free solution procedure should be performed");
        prm.add_parameter("advec diff do print l2norm",
                          advec_diff.do_print_l2norm,
                          "Defines if the l2norm of the advected field should be printed).");
      }
      prm.leave_subsection();

      /*
       *   levelset
       */
      prm.enter_subsection("levelset");
      {
        prm.add_parameter(
          "ls artificial diffusivity",
          ls.artificial_diffusivity,
          "Defines the artificial diffusivity for the level set transport equation");

        prm.add_parameter("ls do reinitialization",
                          ls.do_reinitialization,
                          "Defines if reinitialization of level set function is activated");
        prm.add_parameter("ls theta",
                          ls.theta,
                          "Sets the theta value for the time stepping scheme (0=explicit euler; "
                          "1=implicit euler; 0.5=Crank-Nicholson");

        prm.add_parameter("ls start time",
                          ls.start_time,
                          "Defines the start time for the solution of the levelset problem");
        prm.add_parameter("ls end time",
                          ls.end_time,
                          "Sets the end time for the solution of the levelset problem");
        prm.add_parameter(
          "ls time step size",
          ls.time_step_size,
          "Sets the step size for time stepping. For non-uniform "
          "time stepping, this parameter determines the size of the first time step.");
        prm.add_parameter("ls enable CFL condition",
                          ls.enable_CFL_condition,
                          "Enables to dynamically adapt the time step to meet the CFL condition"
                          " in case of explicit time integration (theta=0)");
        prm.add_parameter("ls do print l2norm",
                          ls.do_print_l2norm,
                          "Defines if the l2norm of the levelset result should be printed)");
        prm.add_parameter(
          "ls do matrix free",
          ls.do_matrix_free,
          "Set this parameter if a matrix free solution procedure should be performed");
      }
      prm.leave_subsection();

      /*
       *   reinitialization
       */
      prm.enter_subsection("reinitialization");
      {
        prm.add_parameter("reinit max n steps",
                          reinit.max_n_steps,
                          "Sets the maximum number of reinitialization steps");
        prm.add_parameter(
          "reinit constant epsilon",
          reinit.constant_epsilon,
          "Defines the length parameter of the level set function to be constant and"
          "not to dependent on the mesh size (default: -1.0 i.e. grid size dependent"
          "which can be controlled by reinit_epsilon_scale_factor");
        prm.add_parameter(
          "reinit scale factor epsilon",
          reinit.scale_factor_epsilon,
          "Defines the scaling factor of the diffusion parameter in the reinitialization "
          "equation; the scaling factor is multipled by the mesh size (default: 0.5 i.e. eps=0.5*h_min");
        prm.add_parameter(
          "reinit dtau",
          reinit.dtau,
          "Defines the time step size of the reinitialization to be constant and"
          "not to dependent on the mesh size (default: -1.0 i.e. grid size dependent");
        prm.add_parameter("reinit modeltype",
                          reinit.modeltype,
                          "Sets the type of reinitialization model that should be used.");
        prm.add_parameter(
          "reinit do matrix free",
          reinit.do_matrix_free,
          "Set this parameter if a matrix free solution procedure should be performed");
        prm.add_parameter(
          "reinit do print l2norm",
          reinit.do_print_l2norm,
          "Defines if the l2norm of the reinitialization result should be printed)");
      }
      prm.leave_subsection();
      /*
       *   normal vector
       */
      prm.enter_subsection("normal vector");
      {
        prm.add_parameter(
          "normal vec damping scale factor",
          normal_vec.damping_scale_factor,
          "normal vector computation: damping = cell_size * normal_vec_damping_scale_factor");
        prm.add_parameter(
          "normal vec do matrix free",
          normal_vec.do_matrix_free,
          "Set this parameter if a matrix free solution procedure should be performed");
        prm.add_parameter("normal vec do print l2norm",
                          normal_vec.do_print_l2norm,
                          "Defines if the l2norm of the normal vector result should be printed)");
      }
      prm.leave_subsection();
      /*
       *   curvature
       */
      prm.enter_subsection("curvature");
      {
        prm.add_parameter("curv damping scale factor",
                          curv.damping_scale_factor,
                          "curvature computation: damping = cell_size * curv_damping_scale_factor");
        prm.add_parameter(
          "curv do matrix free",
          curv.do_matrix_free,
          "Set this parameter if a matrix free solution procedure should be performed");
        prm.add_parameter("curv do print l2norm",
                          curv.do_print_l2norm,
                          "Defines if the l2norm of the curvature result should be printed)");
      }
      prm.leave_subsection();
      /*
       *   flow
       */
      prm.enter_subsection("flow");
      {
        prm.add_parameter("flow density",
                          flow.density,
                          "density of the flow field");
        prm.add_parameter("flow density difference",
                          flow.density_difference,
                          "density difference of the two-phase flow field");
        prm.add_parameter("flow viscosity",
                          flow.viscosity,
                          "viscosity of the flow field");
        prm.add_parameter("flow viscosity difference",
                          flow.viscosity_difference,
                          "viscosity difference of the two-phase flow field");
        prm.add_parameter("flow density",
                          flow.density,
                          "density of the flow field");
        prm.add_parameter("flow surface tension coefficient",
                          flow.surface_tension_coefficient,
                          "constant coefficient for calculating surface tension");
        prm.add_parameter("flow solver type",
                          flow.solver_type,
                          "solver type of the flow problem");                          
        prm.add_parameter("flow start time",
                          flow.start_time,
                          "Defines the start time for the solution of the levelset problem");
        prm.add_parameter("flow end time",
                          flow.end_time,
                          "Sets the end time for the solution of the levelset problem");
        prm.add_parameter("flow time step size",
                           flow.time_step_size,
                           "Sets the step size for time stepping. For non-uniform "
                           "time stepping, this parameter determines the size of the first "
                           "time step.");
        prm.add_parameter("flow max n steps",
                          flow.max_n_steps,
                          "Sets the maximum number of flow steps");
      }
      prm.leave_subsection();
      /*
       *   paraview
       */
      prm.enter_subsection("paraview");
      {
        prm.add_parameter("paraview do output",
                          paraview.do_output,
                          "boolean for producing paraview output files");
        prm.add_parameter("paraview filename",
                          paraview.filename,
                          "Sets the base name for the paraview file output.");
        prm.add_parameter("paraview write frequency",
                          paraview.write_frequency,
                          "every n timestep that should be written");
        prm.add_parameter("paraview do initial state",
                          paraview.do_initial_state,
                          "boolean for writing the initial state into the paraview output file");
        prm.add_parameter(
          "paraview print levelset",
          paraview.print_levelset,
          "boolean for writing the levelset variable into the paraview output file");
        prm.add_parameter(
          "paraview print normal vector",
          paraview.print_normal_vector,
          "boolean for writing the computed normalvector into the paraview output file");
        prm.add_parameter(
          "paraview print curvature",
          paraview.print_curvature,
          "boolean for writing the computed curvature into the paraview output file");
        prm.add_parameter(
          "paraview print advection",
          paraview.print_advection,
          "boolean for writing the computed advection into the paraview output file");
        prm.add_parameter("paraview print exactsolution",
                          paraview.print_exactsolution,
                          "boolean for writing the exact solution into the paraview output file");
        prm.add_parameter("paraview print boundary id",
                          paraview.print_boundary_id,
                          "boolean for printing a vtk-file with the boundary id");
        prm.add_parameter("paraview n digits timestep",
                          paraview.n_digits_timestep,
                          "number of digits for the frame number of the vtk-file.");
        prm.add_parameter("paraview n groups",
                          paraview.n_digits_timestep,
                          "number of parallel written vtk-files.");
      }
      prm.leave_subsection();

      /*
       *   output
       */
      prm.enter_subsection("output");
      {
        prm.add_parameter(
          "do walltime",
          output.do_walltime,
          "this flag enables the output of wall times (should be disabled if a test file is prepared)");
        prm.add_parameter(
          "do compute error",
          output.do_compute_error,
          "this flag enables the computation of the error compared to a given analytical solution.");
        prm.add_parameter("do compute volume output",
                          output.do_compute_volume_output,
                          "boolean for computing the phase volumes");
        prm.add_parameter("filename volume output",
                          output.filename_volume_output,
                          "Sets the base name for the volume fraction file output.");
      }
      prm.leave_subsection();
    }

    ParameterHandler prm;

    BaseData<number>               base;
    AdaptiveMeshingData<number>    amr;
    LevelSetData<number>           ls;
    ReinitializationData<number>   reinit;
    AdvectionDiffusionData<number> advec_diff;
    FlowData<number>               flow;
    NormalVectorData<number>       normal_vec;
    CurvatureData<number>          curv;
    ParaviewData<number>           paraview;
    OutputData<number>             output;
    Flow::AdafloWrapperParameters  adaflo_params;
  };


} // namespace MeltPoolDG
