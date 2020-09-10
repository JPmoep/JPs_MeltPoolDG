#pragma once
#include <deal.II/base/parameter_handler.h>
// c++
#include <fstream>
#include <iostream>

namespace MeltPoolDG
{

using namespace dealii;

template <typename number=double>
struct Parameters
{
  void 
  process_parameters_file(const std::string &parameter_filename)
  {
    
    ParameterHandler prm;
    add_parameters(prm);
    
    check_for_file(parameter_filename, prm);

    std::ifstream file;
    file.open(parameter_filename);
    
    if(parameter_filename.substr(parameter_filename.find_last_of(".") + 1) == "json") 
      prm.parse_input_from_json(file, true);
    else if(parameter_filename.substr(parameter_filename.find_last_of(".") + 1) == "prm") 
      prm.parse_input(parameter_filename);
    else
      AssertThrow(false, ExcMessage("Parameterhandler cannot handle current file ending"));
    prm.print_parameters(std::cout,
                         ParameterHandler::OutputStyle::Text);
  }
  
  void check_for_file (const std::string &parameter_filename,
                       ParameterHandler  & /*prm*/) const
  {
    std::ifstream parameter_file (parameter_filename.c_str());

    if (!parameter_file)
      {
        parameter_file.close ();

        std::ostringstream message;
        message << "Input parameter file <" << parameter_filename
                << "> not found. Please make sure the file exists!"
                << std::endl;

        AssertThrow (false, ExcMessage (message.str().c_str()));
      }
  }
  
  void 
  add_parameters(ParameterHandler& prm)
  {
    /*
     *    general
     */
    prm.enter_subsection("general");
    {
      prm.add_parameter("problem name", 
                         problem_name, 
                        "Sets the base name for the problem that should be solved."
                        );
    }
    prm.leave_subsection();
    /*
     *    spatial domain
     */
    prm.enter_subsection("spatial domain");
    {
      prm.add_parameter("dimension", 
                         dimension,
                        "Defines the dimension of the problem (default value=2)"
                         );
      prm.add_parameter("global refinements",
                         global_refinements,
                        "Defines the number of initial global refinements (default value=1)"
                         );
    }
    prm.leave_subsection();
    /*
     *   levelset 
     */
    prm.enter_subsection("advection diffusion");
    {
      prm.add_parameter("advec diff diffusivity",
                         advec_diff_diffusivity,
                        "Defines the diffusivity for the advection diffusion equation "
                        );
      prm.add_parameter("advec diff theta", 
                         advec_diff_theta,
                        "Sets the theta value for the time stepping scheme: 0=explicit euler; "
                        "1=implicit euler; 0.5=Crank-Nicholson; (default=0.5)"
                        );
      prm.add_parameter("advec diff start time", 
                         advec_diff_start_time,
                        "Defines the start time for the solution of the levelset problem (default=0.0)"
                        );
      prm.add_parameter("advec diff end time", 
                         advec_diff_end_time,
                        "Sets the end time for the solution of the advection diffusion problem (default=1.0)"
                        );
      prm.add_parameter("advec diff time step size", 
                         advec_diff_time_step_size,
                        "Sets the step size for time stepping (default=0.01). For non-uniform "
                        "time stepping, this parameter determines the size of the first time step."
                        );
      prm.add_parameter("advec diff do matrixfree",
                         advec_diff_do_matrixfree,
                        "Set this parameter if a matrixfree solution procedure should be performed (default=false)");
      prm.add_parameter("advec diff do print l2norm",
                         advec_diff_do_print_l2norm,
                        "Defines if the l2norm of the advected field should be printed) "
                        "(default=false)");
    }
    prm.leave_subsection();
    
    /*
     *   levelset 
     */
    prm.enter_subsection("levelset");
    {
      prm.add_parameter("ls artificial diffusivity",
                         ls_artificial_diffusivity,
                        "Defines the artificial diffusivity for the level set transport equation"
                        );

      prm.add_parameter("ls do reinitialization",
                         ls_do_reinitialization,
                        "Defines if reinitialization of level set function is activated (default=false)"
                       );
      prm.add_parameter("ls theta", 
                         ls_theta,
                        "Sets the theta value for the time stepping scheme (0=explicit euler; "
                        "1=implicit euler; 0.5=Crank-Nicholson (default)"
                        );

      prm.add_parameter("ls start time", 
                         ls_start_time,
                        "Defines the start time for the solution of the levelset problem (default=0.0)"
                        );
      prm.add_parameter("ls end time", 
                         ls_end_time,
                        "Sets the end time for the solution of the levelset problem (default=1.0)"
                        );
      prm.add_parameter("ls time step size", 
                         ls_time_step_size,
                        "Sets the step size for time stepping (default=0.01). For non-uniform "
                        "time stepping, this parameter determines the size of the first time step."
                        );
      prm.add_parameter("ls enable CFL condition", 
                         ls_enable_CFL_condition,
                        "Enables to dynamically adapt the time step to meet the CFL condition"
                        " in case of explicit time integration (theta=0) (default=false)"
                        );
      prm.add_parameter("ls do print l2norm",
                         ls_do_print_l2norm,
                        "Defines if the l2norm of the levelset result should be printed) "
                        "(default=false)");
    }
    prm.leave_subsection();
    
    /*
     *   reinitialization 
     */
    prm.enter_subsection("reinitialization");
    {
      prm.add_parameter("reinit max n steps", 
                         reinit_max_n_steps,
                        "Sets the maximum number of reinitialization steps (default value=5)"); 
      prm.add_parameter("reinit constant epsilon",
                         reinit_constant_epsilon,             
                        "Defines the length parameter of the level set function to be constant and"
                        "not to dependent on the mesh size (default: -1.0 i.e. grid size dependent");
      prm.add_parameter("reinit dtau",
                         reinit_dtau,
                        "Defines the time step size of the reinitialization to be constant and"
                        "not to dependent on the mesh size (default: -1.0 i.e. grid size dependent");
      prm.add_parameter("reinit modeltype",
                         reinit_modeltype,
                        "Sets the type of reinitialization model that should be used (default=olsson2007)"
                        "This string is converted to an enum value."); 
      prm.add_parameter("reinit do matrixfree",
                         reinit_do_matrixfree,
                        "Set this parameter if a matrixfree solution procedure should be performed (default=false)");
      prm.add_parameter("reinit do print l2norm",
                         reinit_do_print_l2norm,
                        "Defines if the l2norm of the reinitialization result should be printed) (default: false)");
    }
    prm.leave_subsection();
    
    /*
     *   paraview
     */
    prm.enter_subsection("paraview");
    {
      prm.add_parameter ("paraview do output", 
                          paraview_do_output,
                         "boolean for producing paraview output files (default=false)");
      prm.add_parameter ("paraview filename", 
                          paraview_filename,
                         "Sets the base name for the paraview file output. (default=solution)");
      prm.add_parameter ("paraview write frequency", 
                          paraview_write_frequency,
                         "every n timestep that should be written (default=1)");
      prm.add_parameter ("paraview do initial state",
                          paraview_do_initial_state,
                         "boolean for writing the initial state into the paraview output file");
      prm.add_parameter ("paraview print levelset",
                          paraview_print_levelset,
                         "boolean for writing the levelset variable into the paraview output file");
      prm.add_parameter ("paraview print normal vector", 
                          paraview_print_normal_vector,
                         "boolean for writing the computed normalvector into the paraview output file");
      prm.add_parameter ("paraview print curvature", 
                          paraview_print_curvature,
                         "boolean for writing the computed curvature into the paraview output file");
      prm.add_parameter ("paraview print advection", 
                          paraview_print_advection,
                         "boolean for writing the computed advection into the paraview output file");
      prm.add_parameter ("paraview print exactsolution", 
                          paraview_print_exactsolution,
                         "boolean for writing the exact solution into the paraview output file");
    }
    prm.leave_subsection();
    
    /*
     *   output
     */
    prm.enter_subsection("output");
    {
      prm.add_parameter ("output walltime", 
                          output_walltime,
                         "this flag enables the output of wall times (should be disabled if a test file is prepared)");
      prm.add_parameter ("do compute error", 
                          do_compute_error,
                         "this flag enables the computation of the error compared to a given analytical solution.");
      prm.add_parameter ("compute volume output", 
                          compute_volume_output,
                         "boolean for computing the phase volumes");
      prm.add_parameter ("filename volume output", 
                          filename_volume_output,
                         "Sets the base name for the volume fraction file output.");
    }
    prm.leave_subsection();
  }

  // general
  std::string         problem_name              = "advection_diffusion";
  // spatial
  unsigned int        dimension                 = 2;
  unsigned int        global_refinements        = 1;
  //unsigned int        adaptive_refinements;
  
  // level set specific parameters
  bool                ls_do_reinitialization    = false;
  number              ls_artificial_diffusivity = 0.0;
  number              ls_theta                  = 0.5;
  number              ls_start_time             = 0.0;
  number              ls_end_time               = 1.0;
  number              ls_time_step_size         = 0.01;
  bool                ls_enable_CFL_condition   = false; 
  bool                ls_do_print_l2norm        = false;

  // reinitialization specific parameters
  unsigned int        reinit_max_n_steps        = 5;
  number              reinit_constant_epsilon   = -1.0;
  number              reinit_dtau               = -1.0;
  unsigned int        reinit_modeltype          = 1;  //@ readability could be improved by using a string variable
  bool                reinit_do_matrixfree      = false;
  bool                reinit_do_print_l2norm    = false;

  // advection diffusion specific parameters
  number              advec_diff_diffusivity     = 0.0;
  number              advec_diff_theta           = 0.5;
  number              advec_diff_start_time      = 0.0;
  number              advec_diff_end_time        = 1.0;
  number              advec_diff_time_step_size  = 0.01;
  bool                advec_diff_do_print_l2norm = false;
  bool                advec_diff_do_matrixfree   = false;

  // normal vector    specific parameters
  // @todo
  
  // curvature        specific parameters
  // @todo
  
  // paraview options
  bool                paraview_do_output            = false;
  std::string         paraview_filename             = "solution";
  int                 paraview_write_frequency      = 1;
  bool                paraview_do_initial_state     = true;
  bool                paraview_print_levelset       = true;
  bool                paraview_print_normal_vector  = false;
  bool                paraview_print_curvature      = false;
  bool                paraview_print_advection      = false;
  bool                paraview_print_exactsolution  = false;
  
  // output options
  bool                output_walltime               = 0;
  bool                do_compute_error              = 0;
  bool                compute_volume_output         = 0;
  std::string         filename_volume_output        = "my_volumes.txt";
};

} // namespace MeltPoolDG
