#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

#include "book/definitions.hpp"

Tensor_1D softmax(const Tensor_1D &z)
{
    const Tensor_0D m = z.maximum();
    auto normalized = z - z.constant(m(0));
    auto expo = normalized.exp();
    const Tensor_0D expo_sums = expo.sum();
    Tensor_1D result = expo / expo.constant(expo_sums(0));

    return result;
}

int main(int, char **)
{
    Tensor_2D input(8, 3);
    input.setValues({
        {0.1, 1., -2.},{10., 2., 5.},{5., -5., 0.},{2., 3., 2.},
        {100., 1000., -500.},{3., 3., 3.},{-1, 1., -1.},{-11., -0.2, -.1}
    });

    const int batch_size = input.dimension(0);
    const int output_size = input.dimension(1);

    DimArray<2> extent = {1, output_size};
    Eigen::array<int, 1> reshape_dim({output_size});
    for (int i = 0; i < batch_size; i++) {
        DimArray<2> offset = {i, 0};
        Tensor_1D row = input.slice(offset, extent).reshape(reshape_dim);

        Tensor_1D output = softmax(row);

        std::cout << "softmax([" << row << "]): [" << output << "]\n\n";
    }

    return 0;
}