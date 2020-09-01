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
  prm.enter_subsection ("spatial domain");
  {
      prm.declare_entry ("dimension", "2", Patterns::Integer(),
                         "Defines the dimension of the problem (default value=2)");
      prm.declare_entry ("global refinements","1",Patterns::Integer(),
                         "Defines the number of initial global refinements (default value=1)");
      prm.declare_entry ("do matrix free", "0", Patterns::Integer(),
                         "this flag enables whether a matrix free simulation will be activated (where available)");
      //prm.declare_entry ("adaptive refinements","0",Patterns::Integer(),
                         //"Defines the number of adaptive refinements.)
  }
  prm.leave_subsection();
  
  prm.enter_subsection ("level set equation");
  {
      prm.declare_entry    ("level set degree", "1", Patterns::Integer(),
                            "Sets the degree for the level set function (default value=1)"); 
      prm.declare_entry    ("artificial diffusivity","0.0",Patterns::Double(),
                            "Defines the artificial diffusivity for the level set transport equation");
      prm.declare_entry    ("activate reinitialization","0",Patterns::Integer(),
                            "Defines if reinitialization of level set function is activated (default=false)");
      prm.declare_entry    ("max reinitialization steps","2",Patterns::Integer(),
                            "Defines the maximum number of reinitialization steps (default=2)");
  }
  prm.leave_subsection();

  prm.enter_subsection ("time stepping");
  {
      prm.declare_entry ("theta", "0.5", Patterns::Double(),
                         "Sets the theta value for the time stepping scheme (0=explicit euler; 1=implicit euler; 0.5=Crank-Nicholson (default)");
      prm.declare_entry ("start time", "0.", Patterns::Double(),
                         "Sets the start time for the simulation");
      prm.declare_entry ("end time", "1.", Patterns::Double(),
                         "Sets the final time for the simulation");
      prm.declare_entry ("time step size", "1.e-2", Patterns::Double(),
                         "Sets the step size for time stepping. For non-uniform "
                         "time stepping, this sets the size of the first time "
                         "step.");
      prm.declare_entry ("enable CFL condition", "0", Patterns::Integer(),
                         "Enables to dynamically adapt the time step to the current"
                         " mesh size");
  }
  prm.leave_subsection();
  prm.enter_subsection ("output");
  {
      prm.declare_entry ("output walltime", "0", Patterns::Integer(),
                         "this flag enables the output of wall times (should be disabled if a test file is prepared)");
      prm.declare_entry ("output norm levelset", "0", Patterns::Integer(),
                         "this flag enables the output of the norm of the dof vector (should be ENABLED if a test file is prepared)");
      prm.declare_entry ("compute paraview output", "0", Patterns::Integer(),
                         "boolean for producing paraview output files");
      prm.declare_entry ("filename paraview output", "solution", Patterns::Anything(),
                         "Sets the base name for the paraview file output.");
      prm.declare_entry ("compute volume output", "0", Patterns::Integer(),
                         "boolean for computing the phase volumes");
      prm.declare_entry ("filename volume output", "volumes", Patterns::Anything(),
                         "Sets the base name for the volume fraction file output.");
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

  prm.enter_subsection("spatial domain");
      dimension          = prm.get_integer ("dimension");
      global_refinements = prm.get_integer ("global refinements");
      do_matrix_free     = prm.get_integer ("do matrix free");
      //adaptive_refinements = prm.get_integer ("adaptive refinements");
  prm.leave_subsection ();

  prm.enter_subsection("level set equation");
      levelset_degree        =    prm.get_integer("level set degree");
      AssertThrow (levelset_degree >= 1, ExcNotImplemented());
      artificial_diffusivity =     prm.get_double("artificial diffusivity");
      activate_reinitialization = prm.get_integer("activate reinitialization");
      max_n_reinit_steps     =    prm.get_integer("max reinitialization steps");
  prm.leave_subsection ();

  prm.enter_subsection ("time stepping");
      theta                = (prm.get_double ("theta"));
      start_time           = (prm.get_double ("start time"));
      end_time             = (prm.get_double ("end time"));
      time_step_size       = (prm.get_double ("time step size"));
      AssertThrow (time_step_size>= 0.0, ExcNotImplemented());
      enable_CFL_condition = (prm.get_double ("enable CFL condition"));
  prm.leave_subsection ();
  prm.enter_subsection ("output");
      output_walltime =         prm.get_integer("output walltime");
      output_norm_levelset =    prm.get_integer("output norm levelset");
      compute_paraview_output = prm.get_integer("compute paraview output");
      filename_paraview_output =        prm.get("filename paraview output");
      compute_volume_output =   prm.get_integer("compute volume output");
      filename_volume_output =          prm.get("filename volume output");
  prm.leave_subsection ();
}

void LevelSetParameters::print_parameters()
{
    std::cout << "+----------------------------------------"                 << std::endl;
    std::cout << "|       input protocol                  |"                 << std::endl;
    std::cout << "+----------------------------------------"                 << std::endl;
    std::cout << "| dimension                 " << dimension                 << std::endl;                   
    std::cout << "| global_refinements        " << global_refinements        << std::endl;                   
    std::cout << "| do_matrix_free            " << do_matrix_free            << std::endl;                   
    std::cout << "| levelset_degree           " << levelset_degree           << std::endl;                   
    std::cout << "| artificial_diffusivity    " << artificial_diffusivity    << std::endl;                   
    std::cout << "| activate_reinitialization " << activate_reinitialization << std::endl;                   
    std::cout << "| max_n_reinit_steps        " << max_n_reinit_steps        << std::endl;                   
    std::cout << "| ------------ time stepping -------------"                << std::endl;    
    std::cout << "| theta                     " << theta                     << std::endl;                   
    std::cout << "| start_time                " << start_time                << std::endl;                   
    std::cout << "| end_time                  " << end_time                  << std::endl;                   
    std::cout << "| time_step_size            " << time_step_size            << std::endl;
    std::cout << "| enable_CFL_condition      " << enable_CFL_condition      << std::endl;
    std::cout << "| --------------- output ----------------"                 << std::endl;    
    std::cout << "| output_walltime           " << output_walltime           << std::endl;                   
    std::cout << "| output_norm_levelset      " << output_norm_levelset      << std::endl;                   
    std::cout << "| compute_paraview_output   " << compute_paraview_output   << std::endl;                   
    std::cout << "| filename_paraview_output  " << filename_paraview_output  << std::endl;                   
    std::cout << "| compute_volume_output     " << compute_volume_output     << std::endl;                   
    std::cout << "| filename_volume_output    " << filename_volume_output    << std::endl;                   
    std::cout << "+----------------------------------------"                 << std::endl;

}

