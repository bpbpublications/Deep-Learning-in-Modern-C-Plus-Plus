#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

#include "book/definitions.hpp"

Tensor_2D softmax_2D(const Tensor_2D &z)
{

    auto dimensions = z.dimensions();

    int batch_size = dimensions.at(0);
    int instances_size = dimensions.at(1);

    // Getting the maximum for each instance.
    // Note that this operation reduces 1 dimension
    DimArray<1> depth_dim({1});
    auto z_max = z.maximum(depth_dim); 

    // Getting the max array as an 2-rank tensor
    DimArray<2> reshape_dim({batch_size, 1});
    auto max_reshaped = z_max.reshape(reshape_dim); 

    // Broadcasting max 
    DimArray<2> bcast({1, instances_size});
    auto max_values = max_reshaped.broadcast(bcast);

    // Normalizing the input
    auto normalized = z - max_values;

    // calculating softmax
    auto expo = normalized.exp();
    auto expo_sums = expo.sum(depth_dim);
    auto sums_reshaped = expo_sums.reshape(reshape_dim);
    auto sums = sums_reshaped.broadcast(bcast);
    Tensor_2D result = expo / sums;

    return result;
}

int main(int, char **)
{
    Tensor_2D input(8, 3);
    input.setValues({
        {0.1, 1., -2.},{10., 2., 5.},{5., -5., 0.},{2., 3., 2.},
        {100., 1000., -500.},{3., 3., 3.},{-1, 1., -1.},{-11., -0.2, -.1}
    });

    auto output_2D = softmax_2D(input);

    std::cout << "\noutput_2D:\n" << output_2D << "\n\n";

    return 0;
}