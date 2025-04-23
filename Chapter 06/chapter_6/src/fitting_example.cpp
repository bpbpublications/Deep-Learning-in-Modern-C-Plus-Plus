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

    float Cx = ((Tensor_0D)(X.mean()))(0);
    float Cy = ((Tensor_0D)(Y.mean()))(0);

    float best_cost = std::numeric_limits<float>::max();
    const float step = 0.01;
    float a = -4;
    float best_a = a;

    auto S = [&a, &Cx, &Cy](float x) {
        return a * (x - Cx) + Cy;
    };

    int epochs = 0;

    for (; a < 9; a += step) {

        Tensor_2D pred = X.unaryExpr(S);

        float cost = mse(pred, Y);
        if (cost < best_cost) {
            best_cost = cost;
            best_a = a;
        }
        epochs++;

    }

    const float best_b = Cy - best_a*Cx;

    std::cout << "Best configuration is (a, b) = (" << best_a << ", " << best_b << ")" << " with cost " << best_cost << "\n";
    std::cout << "Took " << epochs << " epochs.\n";

    return 0;
}
