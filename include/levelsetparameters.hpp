#pragma once
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <fstream>
#include <iostream>

using namespace dealii;

struct LevelSetParameters
{
  LevelSetParameters ();

  LevelSetParameters (const std::string &parameter_filename);

  static void declare_parameters (ParameterHandler &prm);
  void parse_parameters (const std::string parameter_filename,
                         ParameterHandler &prm);

  void check_for_file (const std::string &parameter_filename,
                       ParameterHandler  &prm) const;
  
  void print_parameters();
  
  // discretization
  unsigned int        dimension;
  unsigned int        global_refinements;
  //unsigned int        adaptive_refinements;
  //bool                use_anisotropic_refinement;

  // level set specific parameters
  unsigned int        levelset_degree;
  double              artificial_diffusivity;
  //double              interface_thickness; // to be calculated from the characteristic mesh size
  bool                activate_reinitialization;
  bool                compute_volume;
  unsigned int        max_n_reinit_steps;

  // time stepping
  double              theta;
  double              start_time;
  double              end_time;
  double              time_step_size;

  // output options
  //
};

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

template <int dim>
class AdvectionField : public TensorFunction<1, dim>
{
    public:
        AdvectionField()
          : TensorFunction<1, dim>()
        {}

        virtual Tensor<1, dim> value(const Point<dim> & p) const override;
};

