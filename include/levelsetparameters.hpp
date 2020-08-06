#pragma once
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

struct LevelSetParameters {
    double timeStep                 ;   
    double maxTime                  ;   
    double theta                    ;  // 0 = explicit euler, 0.5 = Crank-Nicolson, 1.0 = implicit euler
    double diffusivity              ;  // artificial diffusivity
    bool   activateReinitialization ; 
    bool   computeVolume            ; 
    double dirichletBoundaryValue   ; 
    unsigned int levelSetDegree     ; 
    double epsInterface             ;  
    double characteristicMeshSize   ;  
};

using namespace dealii;


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
