#include <iostream>
#include <random>
#include <limits>
#include <unsupported/Eigen/CXX11/Tensor>

#include "book/definitions.hpp"

template <int _RANK>
auto mse(const Tensor<_RANK> &PRED, const Tensor<_RANK> &TRUE) {
    auto diff = TRUE - PRED;
    auto loss = diff.pow(2.);
    TYPE sum = ((Tensor_0D)(loss.sum()))(0);
    TYPE result = sum / PRED.size();
    return result;
}

int main(int, char**)
{
    int seed = 1234;
    std::mt19937 rng(seed);

    auto synthetic_generator = [&rng](int size, float range, float a, float b, float noise) {

        Tensor_2D X(1, size);
        X = X.random() * range;

        Tensor_2D Y(1, size);
        Y.setConstant(b);

        std::normal_distribution<float> normal_distro(0, noise);

        auto random_gen = [&rng, &normal_distro, a](float interceptor, float x) {
            return interceptor + normal_distro(rng) + x*a;
        };
        
        Y = Y.binaryExpr(X, random_gen);
        
        return std::make_pair(X, Y);

    };

    // generating 30 instances, in a range of 0 <= X < 10, with a = 2, b = 3 and noise deviation = 2
    auto [X, Y] = synthetic_generator(30, 10.f, 2.f, 3.f, 2.f);

    Tensor_2D H0 = X.unaryExpr([](float x) {return -1.f*x + 4.f;});
    Tensor_2D H1 = X.unaryExpr([](float x) {return 1.5f*x + 1.f;});
    
    float cost_h0 = mse(H0, Y);
    float cost_h1 = mse(H1, Y);

    std::cout << "Cost(Y, H0): " << cost_h0 << std::endl;
    std::cout << "Cost(Y, H1): " << cost_h1 << std::endl;

    return 0;
}
