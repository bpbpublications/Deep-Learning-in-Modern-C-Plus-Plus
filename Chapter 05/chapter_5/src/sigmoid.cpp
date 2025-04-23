#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

#include "book/definitions.hpp"

float sigmoid(float z)
{
    float result;
    if (z >= 45.f)
        result = 1.f;
    else if (z <= -45.f)
        result = 0.f;
    else
        result = 1.f / (1.f + exp(-z));
    return result;
}

template <int _RANK>
auto sigmoid_activation(const Tensor<_RANK>& Z)
{
    auto result = Z.unaryExpr(std::ref(sigmoid));
    return result;
}

int main(int, char **)
{

    Tensor_2D A(2, 3);
    A.setRandom();

    std::cout << "A:\n\n"
              << A << "\n\n";

    Tensor_2D B = sigmoid_activation(A);

    std::cout << "B:\n\n"
              << B << "\n\n";

    return 0;
}
