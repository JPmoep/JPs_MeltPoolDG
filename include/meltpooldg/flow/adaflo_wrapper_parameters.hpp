/* ---------------------------------------------------------------------
 *
 * Author: Magdalena Schreter, Peter Münch, TUM, October 2020
 *
 * ---------------------------------------------------------------------*/

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

        // declare parameters
        {
          prm_adaflo.enter_subsection("Navier-Stokes");
          prm_adaflo.enter_subsection("adaflo");
          params.declare_parameters(prm_adaflo);
          prm_adaflo.leave_subsection();
          prm_adaflo.leave_subsection();
        }

        // parse parameters
        {
        std::ifstream file;
        file.open(parameter_filename);

        if (parameter_filename.substr(parameter_filename.find_last_of(".") + 1) == "json")
          prm_adaflo.parse_input_from_json(file, true);
        else if (parameter_filename.substr(parameter_filename.find_last_of(".") + 1) == "prm")
          prm_adaflo.parse_input(parameter_filename);
        else
          AssertThrow(false, ExcMessage("Parameterhandler cannot handle current file ending"));
        }
      
        // read parsed parameters
        {
          prm_adaflo.enter_subsection("Navier-Stokes");
          prm_adaflo.enter_subsection("adaflo");
          params.parse_parameters(parameter_filename, prm_adaflo);
          prm_adaflo.leave_subsection();
          prm_adaflo.leave_subsection();
        }
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

