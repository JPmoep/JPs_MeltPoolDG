#pragma once
#include <iostream>

namespace MeltPoolDG
{
  template<typename number=double>
  struct TimeIteratorData
  {
      number start_time             = 0.0;
      number end_time               = 1.0;
      number time_increment         = 0.01;
      unsigned int max_n_time_steps = 1000;
      bool CFL_condition            = false;
  };
  /*
   *  This class provides a simple time stepping routine.
   */
  template<typename number=double>
  class TimeIterator
  {
  public:
      TimeIterator() = default;
      
      void
      initialize(const TimeIteratorData<number>& data_in)
      {
        time_data = data_in;
        current_time = data_in.start_time;
        current_time_increment = data_in.time_increment;
        n_time_steps = 0;
      }

      bool
      is_finished() const
      {
          if (n_time_steps>=time_data.max_n_time_steps)
              return true;
          return current_time>=time_data.end_time;
      }

      number
      get_next_time_increment()
      {
        if (current_time + current_time_increment > time_data.end_time)
          current_time_increment = time_data.end_time-current_time;
        current_time += current_time_increment;
        n_time_steps += 1;
        return current_time_increment;
      }
     
      void
      resize_current_time_increment(const number factor)
      {
        current_time_increment *= factor;
      }
      
      number
      get_current_time() const
      {
        return current_time;
      }
      
      number
      get_current_time_increment() const
      {
        return current_time_increment;
      }
      
      number
      get_current_time_step_number() const
      {
        return n_time_steps;
      }
      
      void
      reset()
      {
        n_time_steps = 0;
        current_time = time_data.start_time;
        current_time_increment = time_data.time_increment;
      }
      
      void
      print_me(std::ostream & pcout) const
      {
        pcout << "      | Time step " << n_time_steps << " at t=" << std::fixed 
              << std::setprecision(5) << current_time << std::endl; 
      }

  
  private:
      TimeIteratorData<number> time_data;
      number                   current_time;
      number                   current_time_increment;
      number                   n_time_steps;
  };

} // MeltPoolDG
