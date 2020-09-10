#pragma once
#include <iostream>

namespace MeltPoolDG
{
  struct TimeIteratorData
  {
      double start_time       = 0.0;
      double end_time         = 1.0;
      double time_increment   = 0.01;
      int max_n_time_steps    = 1000;
      bool CFL_condition      = false;
  };
  /*
   *  This class provides a simple time stepping routine.
   */
  class TimeIterator
  {

  public:
      TimeIterator();
      
      void
      initialize(const TimeIteratorData&);
      
      bool
      is_finished() const;

      double
      get_next_time_increment();
      
      void
      resize_current_time_increment(const double factor);
      
      double
      get_current_time() const;
      
      double
      get_current_time_increment() const;
      
      double
      get_current_time_step_number() const;
      
      void
      print_me(std::ostream & pcout) const;

      void
      reset();
  
  private:
      TimeIteratorData time_data;
      double current_time;
      double current_time_increment;
      double n_time_steps;
  };

} // MeltPoolDG
