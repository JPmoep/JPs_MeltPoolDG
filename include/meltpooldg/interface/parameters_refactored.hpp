#pragma once
//#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parameter_acceptor.h>

#include <fstream>
#include <iostream>

//@ put struct into namespace

namespace MeltPoolDG
{

using namespace dealii;

template <int dim>
class LevelsetParameters : public ParameterAcceptor
{
  public:
    LevelsetParameters()
  : ParameterAcceptor()
  //: ParameterAcceptor("Levelset Parameters/")
  {
    /*
     *    general
     */
    //enter_my_subsection(this->prm);
    //this->prm.enter_subsection("general");
    //{
      ////prm.add_parameter("problem name", 
                         ////problem_name, 
                        ////"Sets the base name for the problem that should be solved."
                        //////this->prm,
                        //////Patterns::Anything()
                        ////);
    //}
    //this->prm.leave_subsection();
    //leave_my_subsection(this->prm);

    /*
     *    spatial domain
     */
    enter_my_subsection(this->prm);
    this->prm.enter_subsection("spatial domain");
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
    this->prm.leave_subsection();
    leave_my_subsection(this->prm);
    
    /*
     *   levelset 
     */
    //enter_my_subsection(this->prm);
    //this->prm.enter_subsection("levelset");
    //{
      //prm.add_parameter("ls artificial diffusivity",
                         //ls_artificial_diffusivity,
                        //"Defines the artificial diffusivity for the level set transport equation"
                        //);

      //prm.add_parameter("ls do reinitialization",
                         //ls_do_reinitialization,
                        //"Defines if reinitialization of level set function is activated (default=false)"
                       //);
      //prm.add_parameter("ls theta", 
                         //ls_theta,
                        //"Sets the theta value for the time stepping scheme (0=explicit euler; "
                        //"1=implicit euler; 0.5=Crank-Nicholson (default)"
                        //);

      //prm.add_parameter("ls start time", 
                         //ls_start_time,
                        //"Defines the start time for the solution of the levelset problem (default=0.0)"
                        //);
      //prm.add_parameter("ls end time", 
                         //ls_end_time,
                        //"Sets the end time for the solution of the levelset problem (default=1.0)"
                        //);
      //prm.add_parameter("ls time step size", 
                         //ls_time_step_size,
                        //"Sets the step size for time stepping (default=0.01). For non-uniform "
                        //"time stepping, this sets the size of the first time "
                        //"step."
                        //);
      //prm.add_parameter("ls enable CFL condition", 
                         //ls_enable_CFL_condition,
                        //"Enables to dynamically adapt the time step to meet the CFL condition"
                        //" in case of explicit time integration (theta=0) (default=false)"
                        //);
      //prm.add_parameter("ls do print l2norm",
                         //ls_do_print_l2norm,
                        //"Defines if the l2norm of the levelset result should be printed) "
                        //"(default=false)");
    //}
    //this->prm.leave_subsection();
    //leave_my_subsection(this->prm);
    
    /*
     *   reinitialization 
     */
    //enter_my_subsection(this->prm);
    //this->prm.enter_subsection("reinitialization");
    //{
      //prm.add_parameter("reinit max n steps", 
                         //reinit_max_n_steps,
                        //"Sets the maximum number of reinitialization steps (default value=5)"); 
      //prm.add_parameter("reinit constant epsilon",
                         //reinit_constant_epsilon,             
                        //"Defines the length parameter of the level set function to be constant and"
                        //"not to dependent on the mesh size (default: -1.0 i.e. grid size dependent");
      //prm.add_parameter("reinit dtau",
                         //reinit_dtau,
                        //"Defines the time step size of the reinitialization to be constant and"
                        //"not to dependent on the mesh size (default: -1.0 i.e. grid size dependent");
      //prm.add_parameter("reinit modeltype",
                         //reinit_modeltype,
                        //"Sets the type of reinitialization model that should be used (default=olsson2007)"
                        //"This string is converted to an enum value."); 
      //prm.add_parameter("reinit do matrixfree",
                         //reinit_do_matrixfree,
                        //"Set this parameter if a matrixfree solution procedure should be performed (default=false)");
      //prm.add_parameter("reinit do print l2norm",
                         //reinit_do_print_l2norm,
                        //"Defines if the l2norm of the reinitialization result should be printed) (default: false)");
    //}
    //this->prm.leave_subsection();
    //leave_my_subsection(this->prm);
    
    /*
     *   paraview
     */
    //enter_my_subsection(this->prm);
    //this->prm.enter_subsection("paraview");
    //{
      //prm.add_parameter ("paraview do output", 
                          //paraview_do_output,
                         //"boolean for producing paraview output files (default=false)");
      ////prm.add_parameter ("paraview filename", 
                          ////paraview_filename,
                         ////"Sets the base name for the paraview file output. (default=solution)");
      //prm.add_parameter ("paraview write frequency", 
                          //paraview_write_frequency,
                         //"every n timestep that should be written (default=1)");
      //prm.add_parameter ("paraview do initial state",
                          //paraview_do_initial_state,
                         //"boolean for writing the initial state into the paraview output file");
      //prm.add_parameter ("paraview print levelset",
                          //paraview_print_levelset,
                         //"boolean for writing the levelset variable into the paraview output file");
      //prm.add_parameter ("paraview print normal vector", 
                          //paraview_print_normal_vector,
                         //"boolean for writing the computed normalvector into the paraview output file");
      //prm.add_parameter ("paraview print curvature", 
                          //paraview_print_curvature,
                         //"boolean for writing the computed curvature into the paraview output file");
      //prm.add_parameter ("paraview print advection", 
                          //paraview_print_advection,
                         //"boolean for writing the computed advection into the paraview output file");
      //prm.add_parameter ("paraview print exactsolution", 
                          //paraview_print_exactsolution,
                         //"boolean for writing the exact solution into the paraview output file");
    //}
    //this->prm.leave_subsection();
    //leave_my_subsection(this->prm);
    
    /*
     *   output
     */
    //enter_my_subsection(this->prm);
    //this->prm.enter_subsection("output");
    //{
      //prm.add_parameter ("output walltime", 
                          //output_walltime,
                         //"this flag enables the output of wall times (should be disabled if a test file is prepared)");
      //prm.add_parameter ("do compute error", 
                          //do_compute_error,
                         //"this flag enables the computation of the error compared to a given analytical solution.");
      //prm.add_parameter ("compute volume output", 
                          //compute_volume_output,
                         //"boolean for computing the phase volumes");
      ////prm.add_parameter ("filename volume output", 
                          ////filename_volume_output,
                         ////"Sets the base name for the volume fraction file output.");
    //}
    //this->prm.leave_subsection();
    //leave_my_subsection(this->prm);
  }

  void print_parameters()
  {

    auto print_parameter = [](std::string name, auto parameter){ std::ostringstream str; 
                                                  str <<  "| " << std::setw(30) << std::left << name
                                                      << std::left << std::setw(30) << parameter << "|" << std::endl;
                                                   return str.str(); };

    auto print_line= [&](){ int length = 0;
                            length = print_parameter("determine length", length ).length();
                            std::ostringstream line; line << "+" << std::string(length-3, '-') << "+" << std::endl;
                            return line.str(); };
    
    auto print_header = [&](std::string name){
                                               std::ostringstream str; 
                                               str << print_line() << "| " << std::setw(10) << " " << std::setw(20) << std::left << name
                                               << std::left << std::setw(30) << " " << "|" << std::endl << print_line();
                                               return str.str(); };
    
    std::cout << print_header("input protocol");
    //std::cout << print_parameter("problem_name",                  problem_name);

    std::cout << print_header("spatial");
    std::cout << print_parameter("dimension",                     dimension);
    std::cout << print_parameter("global_refinements",            global_refinements);
    
    std::cout << print_header("levelset");
    std::cout << print_parameter("ls_artificial_diffusivity",     ls_artificial_diffusivity );
    std::cout << print_parameter("ls_do_reinitialization",        ls_do_reinitialization );
    std::cout << print_parameter("ls_theta",                      ls_theta );
    std::cout << print_parameter("ls_start_time",                 ls_start_time);
    std::cout << print_parameter("ls_end_time",                   ls_end_time);
    std::cout << print_parameter("ls_time_step_size",             ls_time_step_size);
    std::cout << print_parameter("ls_do_print_l2norm",            ls_do_print_l2norm    );

    std::cout << print_header("reinitialization");
    std::cout << print_parameter("reinit_max_n_steps",            reinit_max_n_steps );
    std::cout << print_parameter("reinit_constant_epsilon",       reinit_constant_epsilon );
    std::cout << print_parameter("reinit_dtau",                   reinit_dtau);
    std::cout << print_parameter("reinit_do_print_l2norm",        reinit_do_print_l2norm    );
    std::cout << print_parameter("reinit_do_matrixfree",          reinit_do_matrixfree      );
    std::cout << print_parameter("reinit_modeltype",              reinit_modeltype);

    std::cout << print_header("paraview");
    std::cout << print_parameter("paraview_do_output",            paraview_do_output );
    std::cout << print_parameter("paraview_filename",             paraview_filename );
    std::cout << print_parameter("paraview_print_normal_vector ", paraview_print_normal_vector );
    std::cout << print_parameter("paraview_print_curvature",      paraview_print_curvature);
    std::cout << print_parameter("paraview_print_exactsolution",  paraview_print_exactsolution    );

    std::cout << print_header("output");
    std::cout << print_parameter("output_walltime",               output_walltime  );
    std::cout << print_parameter("do_compute_error",              do_compute_error );
    std::cout << print_parameter("compute_volume_output",         compute_volume_output );
    std::cout << print_parameter("filename_volume_output",        filename_volume_output);
    std::cout << print_line();
  }
  
  // general
  //std::string         problem_name = "levelset";
  // spatial
  unsigned int        dimension = 2;
  unsigned int        global_refinements;
  //unsigned int        adaptive_refinements;

  // level set specific parameters
  double              ls_artificial_diffusivity = 0.0;
  bool                ls_do_reinitialization    = 0;
  double              ls_theta                  = 0.5;
  double              ls_start_time             = 0.0;
  double              ls_end_time               = 1.0;
  double              ls_time_step_size         = 0.01;
  bool                ls_enable_CFL_condition   = 0; 
  bool                ls_do_print_l2norm        = 0;

  // reinitialization specific parameters
  unsigned int        reinit_max_n_steps        = 5;
  double              reinit_constant_epsilon   = -1.0;
  double              reinit_dtau               = -1.0;
  unsigned int        reinit_modeltype          = 1;  //@ readability could be improved by using a string variable
  bool                reinit_do_matrixfree      = 0;
  bool                reinit_do_print_l2norm    = 0;
  
  // normal vector    specific parameters
  // @todo
  
  // curvature        specific parameters
  // @todo
  
  // paraview options
  bool                paraview_do_output            = 0;
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
  std::string         filename_volume_output        = 0;
};

} // namespace MeltPoolDG
