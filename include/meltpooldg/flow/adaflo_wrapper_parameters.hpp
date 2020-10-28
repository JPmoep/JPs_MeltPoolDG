// #pragma once

// #ifdef MELT_POOL_DG_WITH_ADAFLO

// #include <adaflo/parameters.h>
// #include <adaflo/time_stepping.h>

// #include <meltpooldg/interface/parameters.hpp>

// namespace MeltPoolDG
// {
//   namespace Flow
//   {
//     struct
//     AdafloWrapperParameters
//     {
//         template<typename number>
//         AdafloWrapperParameters(const Parameters<number>& mp_params)
//         {
//           params.time_step_scheme = TimeStepping::bdf_2;
//           params.end_time = 8;
//           params.time_step_size_start = 0.02; 
//           params.physical_type   = FlowParameters::PhysicalType::incompressible;
//           params.dimension       = mp_params.base.dimension;
//           params.velocity_degree = mp_params.base.degree;
//           params.viscosity       = mp_params.flow.viscosity;
//           params.linearization   = FlowParameters::Linearization::coupled_implicit_newton;
//           params.max_nl_iteration = 10;
//           params.tol_nl_iteration = 1e-9;
//           params.max_lin_iteration = 30;
//           params.tol_lin_iteration = 1e-5;
//           params.rel_lin_iteration = 1;
//           params.precondition_velocity = FlowParameters::PreconditionVelocity::u_ilu_scalar;
//           params.precondition_pressure = FlowParameters::PreconditionPressure::p_mass_ilu;
//           params.iterations_before_inner_solvers = 30;
//         }

//         const FlowParameters& get_parameters() const
//         {
//           return this->params;
//         }

//         FlowParameters params;
//     };

//   }
// }

// #endif
