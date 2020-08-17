#include <deal.II/base/mpi.h>
#include <levelsetparameters.hpp>

LevelSetParameters::LevelSetParameters()
  :
  dimension(numbers::invalid_unsigned_int)
{
  // do nothing
}

LevelSetParameters::
LevelSetParameters (const std::string &parameter_filename)
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
      //prm.declare_entry ("anisotropic refinement","0",Patterns::Integer(),
                         //"defines whether the mesh should be refined "
                         //"anisotropically in normal direction to the interface, "
                         //"0 means no anisotropy");
      //prm.declare_entry ("adaptive refinements","0",Patterns::Integer(),
                         //"Defines the number of adaptive refinements. Not used "
                         //"in the Navier-Stokes class, but useful in many "
                         //"applications.");
  }
  prm.leave_subsection();
  
  prm.enter_subsection ("level set equation");
  {
      prm.declare_entry    ("level set degree", "1", Patterns::Integer(),
                            "Sets the degree for the level set function (default value=1)"); 
      prm.declare_entry    ("artificial diffusivity","0.0",Patterns::Double(),
                            "Defines the artificial diffusivity for the level set transport equation");
      //prm.declare_entr   y ("interface_thickness","0.0",Patterns::Double(),
                            //"Defines the artificial diffusivity for the level set transport equation");
      prm.declare_entry    ("activate reinitialization","0",Patterns::Integer(),
                            "Defines if reinitialization of level set function is activated (default=false)");
      prm.declare_entry    ("max reinitialization steps","2",Patterns::Integer(),
                            "Defines the maximum number of reinitialization steps (default=2)");
      prm.declare_entry    ("compute volume","0",Patterns::Integer(),
                            "Defines if volume should be computed (default=false)");
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
  }
  prm.leave_subsection();
  prm.enter_subsection ("output");
  {
      prm.declare_entry ("compute paraview output", "0", Patterns::Integer(),
                         "boolena for producing paraview output files");
  }
  prm.leave_subsection();

  //prm.enter_subsection("Output options");
  //prm.declare_entry ("output filename","",Patterns::Anything(),
                     //"Sets the base name for the file output.");
  //prm.declare_entry ("output verbosity","2",Patterns::Integer(),
                     //"Sets the amount of information from the "
                     //"Navier-Stokes solver that is printed to screen. "
                     //"0 means no output at all, and larger numbers mean an "
                     //"increasing amount of output (maximum value: 3). "
                     //"A value of 3 not only includes solver iterations "
                     //"but also details on solution time and some memory "
                     //"statistics.");
  //prm.declare_entry ("output frequency","1",Patterns::Double(),
                     //"defines at with time interface the solution "
                     //"should be written to file (in supported routines)");
  //prm.declare_entry ("output vtk files","0",Patterns::Integer(),
                     //"defines whether to output vtk files with the "
                     //"whole solution field or just collected point data");

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
      //if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        prm.print_parameters(std::cout, ParameterHandler::Text);
      //AssertThrow (false, ExcMessage ("Invalid input parameter file."));
    }

  prm.enter_subsection("spatial domain");
      dimension = prm.get_integer ("dimension");
      global_refinements = prm.get_integer ("global refinements");
      //adaptive_refinements = prm.get_integer ("adaptive refinements");
      //use_anisotropic_refinement = prm.get_integer ("anisotropic refinement") > 0;
  prm.leave_subsection ();

  prm.enter_subsection("level set equation");
      levelset_degree        = prm.get_integer("level set degree");
      artificial_diffusivity = prm.get_double("artificial diffusivity");
      //AssertThrow (levelset_degree >= 1, ExcNotImplemented());
      activate_reinitialization = prm.get_integer("activate reinitialization");
      compute_volume         = prm.get_integer("compute volume");
      max_n_reinit_steps     = prm.get_integer("max reinitialization steps");
  prm.leave_subsection ();

  //prm.enter_subsection("Output options");
  //output_filename = prm.get("output filename");
  //output_verbosity = prm.get_integer("output verbosity");
  //Assert (output_verbosity <= 3, ExcInternalError());
  //output_frequency = prm.get_double("output frequency");
  //print_solution_fields = prm.get_integer("output vtk files");
  //if (print_solution_fields > 2)
    //print_solution_fields = 1;
  //prm.leave_subsection ();

  prm.enter_subsection ("time stepping");
      theta          = (prm.get_double ("theta"));
      start_time     = (prm.get_double ("start time"));
      end_time       =  (prm.get_double ("end time"));
      time_step_size = (prm.get_double ("time step size"));
  prm.leave_subsection ();
  prm.enter_subsection ("output");
      compute_paraview_output = prm.get_integer("compute paraview output");
  prm.leave_subsection ();
}

void LevelSetParameters::print_parameters()
{
    std::cout << "+----------------------------------------"                << std::endl;
    std::cout << "| dimension                 " << dimension                << std::endl;                   
    std::cout << "| global_refinements        " << global_refinements       << std::endl;                   
    std::cout << "| levelset_degree           " << levelset_degree          << std::endl;                   
    std::cout << "| artificial_diffusivity    " << artificial_diffusivity   << std::endl;                   
    std::cout << "| activate_reinitialization " << activate_reinitialization<< std::endl;                   
    std::cout << "| compute_volume            " << compute_volume           << std::endl;                   
    std::cout << "| max_n_reinit_steps        " << max_n_reinit_steps       << std::endl;                   
    std::cout << "| theta                     " << theta                    << std::endl;                   
    std::cout << "| start_time                " << start_time               << std::endl;                   
    std::cout << "| end_time                  " << end_time                 << std::endl;                   
    std::cout << "| time_step_size            " << time_step_size           << std::endl;                   
    std::cout << "| compute_paraview_output   " << compute_paraview_output  << std::endl;                   
    std::cout << "+----------------------------------------"                << std::endl;

}

