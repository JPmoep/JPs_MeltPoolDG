#pragma once
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

namespace MeltPoolDG
{
    using namespace dealii;

    template<int dim>
    struct FieldConditions
    {
        std::shared_ptr<Function<dim>>           initial_field;
        std::shared_ptr<TensorFunction<1,dim>>   advection_field;
        std::shared_ptr<Function<dim>>           exact_solution_field;
    };
} // namespace MeltPoolDG
