#pragma once
#include <iostream>

namespace LevelSetParallel
{

    struct TimeIteratorData
    {
        TimeIteratorData()
            : start_time(0.0)
            , end_time(1.0)
            , time_increment(0.01)
            , max_n_time_steps( 10000 ) // this criteria is stronger than the end_time
            , CFL_condition( false ) // @todo: incorporate CFL condition
        {}

        double start_time;
        double end_time;
        double time_increment;
        int max_n_time_steps;
        bool CFL_condition;
    };
    /*
     *  This class provides a simple time stepping routine.
     */

    class TimeIterator
    {

    public:
        TimeIterator();
        
        void
        initialize(const TimeIteratorData& );
        
        bool
        is_finished();

        double
        get_next_time_increment();
        
        void
        resize_current_time_increment(const double factor);
        
        void
        print_me(std::ostream & pcout);

    
    private:
        TimeIteratorData time_data;
        double current_time;
        double current_time_increment;
        double n_time_steps;
    };

}
