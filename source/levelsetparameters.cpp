#include <deal.II/base/mpi.h>
#include <levelsetparameters.hpp>

LevelSetParameters::LevelSetParameters()
  :
  dimension(numbers::invalid_unsigned_int)
{
  // do nothing
}

void LevelSetParameters::
process_parameters_file(const std::string &parameter_filename)
{
  ParameterHandler prm;
  LevelSetParameters::declare_parameters (prm);
  check_for_file(parameter_filename, prm);
  parse_parameters (parameter_filename, prm);
}


void LevelSetParameters::check_for_file (const std::string &parameter_filename,
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


void LevelSetParameters::declare_parameters (ParameterHandler &prm)
{
  prm.enter_subsection ("general");
  {
      prm.declare_entry ("problem name", "levelset", Patterns::Anything(),
                         "Sets the base name for the problem that should be solved.");
  }
  prm.leave_subsection();
  prm.enter_subsection ("spatial domain");
  {
      prm.declare_entry ("dimension", "2", Patterns::Integer(),
                         "Defines the dimension of the problem (default value=2)");
      prm.declare_entry ("global refinements","1",Patterns::Integer(),
                         "Defines the number of initial global refinements (default value=1)");
      //prm.declare_entry ("do matrix free", "0", Patterns::Integer(),
                         //"this flag enables whether a matrix free simulation will be activated (where available)");
      //prm.declare_entry ("adaptive refinements","0",Patterns::Integer(),
                         //"Defines the number of adaptive refinements.)
  }
  prm.leave_subsection();
  
  prm.enter_subsection ("levelset");
  {
      //prm.declare_entry    ("level set degree", "1", Patterns::Integer(),
                            //"Sets the degree for the level set function (default value=1)"); 
      prm.declare_entry    ("ls artificial diffusivity","0.0",Patterns::Double(),
                            "Defines the artificial diffusivity for the level set transport equation");
      prm.declare_entry    ("ls do reinitialization","0",Patterns::Integer(),
                            "Defines if reinitialization of level set function is activated (default=false)");
      prm.declare_entry ("ls theta", "0.5", Patterns::Double(),
                         "Sets the theta value for the time stepping scheme (0=explicit euler; 1=implicit euler; 0.5=Crank-Nicholson (default)");
      prm.declare_entry ("ls start time", "0.", Patterns::Double(),
                         "Sets the start time for the simulation");
      prm.declare_entry ("ls end time", "1.", Patterns::Double(),
                         "Sets the final time for the simulation");
      prm.declare_entry ("ls time step size", "1.e-2", Patterns::Double(),
                         "Sets the step size for time stepping. For non-uniform "
                         "time stepping, this sets the size of the first time "
                         "step.");
      prm.declare_entry ("ls enable CFL condition", "0", Patterns::Integer(),
                         "Enables to dynamically adapt the time step to the current"
                         " mesh size");
      prm.declare_entry    ("ls do print l2norm","0",Patterns::Integer(),
                            "Defines if the l2norm of the levelset result should be printed) (default: false)");
  }
  prm.leave_subsection();

  prm.enter_subsection ("reinitialization");
  {
      prm.declare_entry    ("reinit max n steps", "1", Patterns::Integer(),
                            "Sets the maximum number of reinitialization steps (default value=5)"); 
      prm.declare_entry    ("reinit constant epsilon","-1.0",Patterns::Double(),
                            "Defines the length parameter of the level set function to be constant and"
                            "not to dependent on the mesh size (default: -1.0 --> grid size dependent");
      prm.declare_entry    ("reinit dtau","-1.0",Patterns::Double(),
                            "Defines the time step size of the reinitialization to be constant and"
                            "not to dependent on the mesh size (default: -1.0 --> grid size dependent");
      prm.declare_entry    ("reinit do print l2norm","0",Patterns::Integer(),
                            "Defines if the l2norm of the reinitialization result should be printed) (default: false)");
      prm.declare_entry    ("reinit do matrixfree","0",Patterns::Integer(),
                            "Set this parameter if a matrixfree solution procedure should be performed (default=false)");
      prm.declare_entry    ("reinit modeltype", "1", Patterns::Integer(),
                            "Sets the type of reinitialization model that should be used (default=olsson2007)"
                            " This string is converted to an enum value."); 
  }
  prm.leave_subsection();

  prm.enter_subsection ("output");
  {
      prm.declare_entry ("output walltime", "0", Patterns::Integer(),
                         "this flag enables the output of wall times (should be disabled if a test file is prepared)");
      prm.declare_entry ("do compute error", "0", Patterns::Integer(),
                         "this flag enables the computation of the error compared to a given analytical solution.");
      prm.declare_entry ("compute volume output", "0", Patterns::Integer(),
                         "boolean for computing the phase volumes");
      prm.declare_entry ("filename volume output", "volumes", Patterns::Anything(),
                         "Sets the base name for the volume fraction file output.");
  }
  prm.leave_subsection();
  prm.enter_subsection ("paraview");
  {
      prm.declare_entry ("paraview do output", "0", Patterns::Integer(),
                         "boolean for producing paraview output files");
      prm.declare_entry ("paraview filename", "solution", Patterns::Anything(),
                         "Sets the base name for the paraview file output.");
      prm.declare_entry ("paraview write frequency", "1", Patterns::Integer(),
                         "every n timestep that should be written");
      prm.declare_entry ("paraview do initial state", "1", Patterns::Integer(),
                         "boolean for writing the initial state into the paraview output file");
      prm.declare_entry ("paraview print levelset", "1", Patterns::Integer(),
                         "boolean for writing the levelset variable into the paraview output file");
      prm.declare_entry ("paraview print normal vector", "0", Patterns::Integer(),
                         "boolean for writing the computed normalvector into the paraview output file");
      prm.declare_entry ("paraview print curvature", "0", Patterns::Integer(),
                         "boolean for writing the computed curvature into the paraview output file");
      prm.declare_entry ("paraview print advection", "0", Patterns::Integer(),
                         "boolean for writing the computed advection into the paraview output file");
      prm.declare_entry ("paraview print exactsolution", "0", Patterns::Integer(),
                         "boolean for writing the exact solution into the paraview output file");
  }
  prm.leave_subsection();
}


void LevelSetParameters::parse_parameters (const std::string parameter_file,
                                           ParameterHandler &prm)
{
  try
    {
      prm.parse_input (parameter_file);
    }
  catch (...)
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        prm.print_parameters(std::cout, ParameterHandler::Text);
     AssertThrow (false, ExcMessage ("Invalid input parameter file."));
    }

  prm.enter_subsection("general");
      problem_name       = prm.get ("problem name");
  prm.leave_subsection ();
  prm.enter_subsection("spatial domain");
      dimension          = prm.get_integer ("dimension");
      global_refinements = prm.get_integer ("global refinements");
      //adaptive_refinements = prm.get_integer ("adaptive refinements");
  prm.leave_subsection ();

  prm.enter_subsection("levelset");
      // @todo: include again when template parameter is removed
      //levelset_degree        =    prm.get_integer("level set degree");
      //AssertThrow (levelset_degree >= 1, ExcNotImplemented());
      ls_artificial_diffusivity =        prm.get_double("ls artificial diffusivity");
      ls_do_reinitialization =           prm.get_integer("ls do reinitialization");
      ls_do_print_l2norm  =              prm.get_integer("ls do print l2norm");
      ls_theta                =          prm.get_double("ls theta");
      ls_start_time           =          prm.get_double("ls start time");
      ls_end_time             =          prm.get_double("ls end time");
      ls_time_step_size       =          prm.get_double("ls time step size");
      //ls_do_matrix_free     = prm.get_integer ("ls do matrix free");
      AssertThrow (ls_time_step_size>= 0.0, ExcNotImplemented());
      // @todo: a critical time step according to the CFL condition should be calculated 
      // when theta=0 (explicit euler)
      //enable_CFL_condition = (prm.get_double ("enable CFL condition"));
  prm.leave_subsection ();

  prm.enter_subsection("reinitialization");
      reinit_max_n_steps      =      prm.get_integer("reinit max n steps");
      reinit_constant_epsilon =       prm.get_double("reinit constant epsilon");
      reinit_dtau             =       prm.get_double("reinit dtau");
      reinit_do_print_l2norm  =      prm.get_integer("reinit do print l2norm");
      reinit_do_matrixfree    =      prm.get_integer("reinit do matrixfree");
      reinit_modeltype        =      prm.get_integer("reinit modeltype");
  prm.leave_subsection();
  prm.enter_subsection ("output");
      output_walltime =              prm.get_integer("output walltime");
      do_compute_error =             prm.get_integer("do compute error");
      compute_volume_output =        prm.get_integer("compute volume output");
      filename_volume_output =               prm.get("filename volume output");
  prm.leave_subsection ();
  prm.enter_subsection ("paraview");
      paraview_do_output =           prm.get_integer("paraview do output");
      paraview_filename  =                   prm.get("paraview filename");
      paraview_write_frequency =     prm.get_integer("paraview write frequency");
      paraview_do_initial_state =    prm.get_integer("paraview do initial state");
      paraview_print_levelset =      prm.get_integer("paraview print levelset");
      paraview_print_normal_vector = prm.get_integer("paraview print normal vector");
      paraview_print_curvature =     prm.get_integer("paraview print curvature");
      paraview_print_advection =     prm.get_integer("paraview print advection");
      paraview_print_exactsolution = prm.get_integer("paraview print exactsolution");
  prm.leave_subsection ();

}

void LevelSetParameters::print_parameters()
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
  std::cout << print_parameter("problem_name",                  problem_name);

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

