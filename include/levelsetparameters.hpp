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

using namespace dealii;

struct LevelSetParameters
{
  LevelSetParameters ();

  void process_parameters_file(const std::string &parameter_filename);

  static void declare_parameters (ParameterHandler &prm);
  void parse_parameters (const std::string parameter_filename,
                         ParameterHandler &prm);

  void check_for_file (const std::string &parameter_filename,
                       ParameterHandler  &prm) const;
  
  void print_parameters();
  
  // general

  // discretization
  std::string         problem_name;
  unsigned int        dimension;
  unsigned int        global_refinements;
  bool                do_matrix_free;
  //unsigned int        adaptive_refinements;

  // level set specific parameters
  unsigned int        levelset_degree;
  double              artificial_diffusivity;
  bool                activate_reinitialization;

  // reinitialization specific parameters
  unsigned int        reinit_max_n_steps;
  double              reinit_constant_epsilon;
  double              reinit_dtau;
  bool                reinit_do_print_l2norm;
  bool                reinit_do_matrixfree;
  unsigned int        reinit_modeltype;       //@ readability could be improved by using a string variable
  //
  // normal vector    specific parameters
  // curvature        specific parameters
  // time stepping
  double              theta;
  double              start_time;
  double              end_time;
  double              time_step_size;
  bool                enable_CFL_condition; 

  // output options
  bool                output_walltime;
  bool                output_norm_levelset;
  bool                do_compute_error;
  bool                compute_volume_output;
  std::string         filename_volume_output;
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
};

namespace LevelSetParallel
{

/* deprecated
template <int dim>
class InitializePhi : public Function<dim>
{
    public:
    InitializePhi()
      : Function<dim>(),
        epsInterface(0.0)
    {}
     double value( const Point<dim> & p,
                   const unsigned int component = 0) const;

     void setEpsInterface(double eps){ this->epsInterface = eps; }

     double getEpsInterface(){ return this->epsInterface; }

    private:
        double epsInterface;

};
*/

template <int dim>
class DirichletCondition : public Function<dim>
{
    public:
        DirichletCondition()
          : Function<dim>()
        {}
        double value(const Point<dim> & p,
                     const unsigned int component = 0) const override;

        void markDirichletEdges(Triangulation<dim>& triangulation_) const;
};

/* deprecated
template <int dim>
class AdvectionField : public TensorFunction<1, dim>
{
    public:
        AdvectionField()
          : TensorFunction<1, dim>()
        {}

        Tensor<1, dim> value(const Point<dim> & p) const override;
};
*/
}
