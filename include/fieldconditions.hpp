#pragma once

namespace LevelSetParallel
{
    using namespace dealii;

    template<int dim>
    struct FieldVariables
    {
        std::shared_ptr<Function<dim>>           initial_field;
        std::shared_ptr<TensorFunction<1,dim>> advection_field;
    };
}
