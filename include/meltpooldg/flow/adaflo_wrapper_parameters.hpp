 #pragma once

 #ifdef MELT_POOL_DG_WITH_ADAFLO

 #include <adaflo/parameters.h>

#include <deal.II/base/parameter_handler.h>

 namespace MeltPoolDG
 {
   namespace Flow
   {
  struct
  AdafloWrapperParameters
  {
      AdafloWrapperParameters() = default;

      void
      parse_parameters(const std::string &parameter_filename) 
      {
        ParameterHandler prm_adaflo;
        params.declare_parameters(prm_adaflo);
        params.parse_parameters(parameter_filename, prm_adaflo);
      }

      const FlowParameters& get_parameters() const
      {
        return this->params;
      }

      FlowParameters params;
  };
   }
 }

 #endif
