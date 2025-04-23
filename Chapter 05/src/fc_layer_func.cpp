#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

#include "book/definitions.hpp"

auto sigmoid = [](float z) {
    float result;
    if (z >= 45.)
        result = 1.;
    else if (z <= -45.)
        result = 0.;
    else
        result = 1. / (1. + exp(-z));
    return result;
};

template <int _RANK >
auto sigmoid_activation(Tensor<_RANK>& Z)
{
	auto result = Z.unaryExpr(std::ref(sigmoid));
    return result;
}

Tensor_1D calc_layer(const Tensor_1D &input, const Tensor_2D &weights, const Tensor_1D &bias) {

    Eigen::array<Eigen::IndexPair<int>, 1> op_dims = {Eigen::IndexPair<int>(0, 0)};
    auto prod = input.contract(weights, op_dims);
    Tensor_1D Z = prod + bias;

    auto result = sigmoid_activation(Z);
    return result;
}

int main(int, char **)
{

    Tensor_1D X(2);
    X.setValues({-1.5, 0.4});

    Tensor_2D W(2, 3);
    W.setValues({{1, 2, 3}, {4, 5, 6}});
    
    Tensor_1D B(3);
    B.setValues({-1, 1, 2});

    std::cout << "X:\n\n"
              << X << "\n\n";
    
    std::cout << "W:\n\n"
              << W << "\n\n";
    
    std::cout << "B:\n\n"
              << B << "\n\n";

    auto R = calc_layer(X, W, B);

    std::cout << "R:\n\n"
              << R << "\n\n";

    return 0;
}
