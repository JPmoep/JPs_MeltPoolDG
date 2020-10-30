 #pragma once

#ifdef MELT_POOL_DG_WITH_ADAFLO
 #include <adaflo/parameters.h>
#endif

#include <deal.II/base/parameter_handler.h>

 namespace MeltPoolDG
 {
   namespace Flow
   {
  struct
  AdafloWrapperParameters
  {
      AdafloWrapperParameters() = default;

 #ifdef MELT_POOL_DG_WITH_ADAFLO
      void
      parse_parameters(const std::string &parameter_filename) 
      {
        ParameterHandler prm_adaflo;
        params.declare_parameters(prm_adaflo);
        params.parse_parameters(parameter_filename, prm_adaflo);
        
        // note: by setting the differences to a non-zero value we force
        //   adaflo to assume that we are running a simulation with variable
        //   coefficients, i.e., it allocates memory for the data structures
        //   variable_densities and variable_viscosities, which are accessed 
        //   during NavierStokesMatrix::begin_densities() and
        //   NavierStokesMatrix::begin_viscosity(). However, we do not actually
        //   use these values, since we fill the density and viscosity 
        //   differently.
        params.density_diff   = 1.0;
        params.viscosity_diff = 1.0;
      }

      const FlowParameters& get_parameters() const
      {
        return this->params;
      }

      FlowParameters params;
 #endif
  };
   }
 }

