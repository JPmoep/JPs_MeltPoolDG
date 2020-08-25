#include <deal.II/base/conditional_ostream.h> 
#include <iostream>
#include "timeiterator.hpp"

namespace LevelSetParallel
{
    using namespace dealii;

    TimeIterator::TimeIterator()
    {
    }

    void
    TimeIterator::initialize(const TimeIteratorData & data_in)
    {
        time_data = data_in;
        current_time = data_in.start_time;
        current_time_increment = data_in.time_increment;
        n_time_steps = 0;
    }

    bool
    TimeIterator::is_finished() 
    {
        if (n_time_steps>=time_data.max_n_time_steps)
            return true;
        return current_time>=time_data.end_time;
    }

    double
    TimeIterator::get_next_time_increment()
    {
        current_time += current_time_increment;
        n_time_steps += 1;
        return current_time_increment;
    }
    
    void
    TimeIterator::resize_current_time_increment(const double factor)
    {
        current_time_increment *= factor;
    }
    //void
    //TimeIterator::print_me(ConditionalOstream pcout)
    //{
    
    //}

    void
    TimeIterator::print_me(std::ostream & pcout)
    {
        //pcout << "      | Time step " << n_time_steps << " at t=" << std::fixed 
              //<< std::setprecision(5) << current_time << std::endl; 
    }
}
