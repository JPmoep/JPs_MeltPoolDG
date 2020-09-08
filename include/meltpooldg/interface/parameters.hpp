#pragma once
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/grid/tria.h>
#include "boundaryconditions.hpp"
#include "fieldconditions.hpp"

#include <fstream>
#include <iostream>

//@ put struct into namespace

namespace MeltPoolDG
{

using namespace dealii;

struct Parameters
{
  Parameters ();

  void process_parameters_file(const std::string &parameter_filename);

  static void declare_parameters (ParameterHandler &prm);
  void parse_parameters (const std::string parameter_filename,
                         ParameterHandler &prm);

  void check_for_file (const std::string &parameter_filename,
                       ParameterHandler  &prm) const;
  
  void print_parameters();
  
  // general
  std::string         problem_name;
  // spatial
  unsigned int        dimension;
  unsigned int        global_refinements;
  //unsigned int        adaptive_refinements;

  // level set specific parameters
  double              ls_artificial_diffusivity;
  bool                ls_do_reinitialization;
  double              ls_theta;
  double              ls_start_time;
  double              ls_end_time;
  double              ls_time_step_size;
  bool                ls_enable_CFL_condition; 
  bool                ls_do_print_l2norm;

  // reinitialization specific parameters
  unsigned int        reinit_max_n_steps;
  double              reinit_constant_epsilon;
  double              reinit_dtau;
  bool                reinit_do_print_l2norm;
  bool                reinit_do_matrixfree;
  unsigned int        reinit_modeltype;       //@ readability could be improved by using a string variable
  
  // normal vector    specific parameters
  // @todo
  
  // curvature        specific parameters
  // @todo
  
  // paraview options
  bool                paraview_do_output;
  std::string         paraview_filename;
  int                 paraview_write_frequency;
  bool                paraview_do_initial_state;
  bool                paraview_print_levelset;
  bool                paraview_print_normal_vector;
  bool                paraview_print_curvature;
  bool                paraview_print_advection;
  bool                paraview_print_exactsolution;
  
  // output options
  bool                output_walltime;
  bool                do_compute_error;
  bool                compute_volume_output;
  std::string         filename_volume_output;
};

} // namespace MeltPoolDG
