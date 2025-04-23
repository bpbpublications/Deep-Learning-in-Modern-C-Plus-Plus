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

        Eigen::Tensor<float, 2> X(1, size);
        X = X.random() * range;

        Eigen::Tensor<float, 2> Y(1, size);
        Y.setConstant(b);

        std::normal_distribution<float> normal_distro(0, noise);

        auto random_gen = [&rng, &normal_distro, a](float interceptor, float x) {
            return interceptor + normal_distro(rng) + x*a;
        };
        
        Y = Y.binaryExpr(X, random_gen);
        
        return std::make_pair(X, Y);

    };

    // generating 30 instances, in a range of 0 <= X < 10, with a = 2, b = 3 and noise deviation = 2
    const auto [X, Y] = synthetic_generator(30, 10.f, 2.f, 3.f, 2.f);

    float Cx = ((Eigen::Tensor<float, 0>)(X.mean()))(0);
    float Cy = ((Eigen::Tensor<float, 0>)(Y.mean()))(0);

    auto cost_calc = [ &Cx, &Cy](float a, auto X, auto Y) {
        auto S = [&a, &Cx, &Cy](float x) {
            return a * (x - Cx) + Cy;
        };
        Eigen::Tensor<float, 2> pred = X.unaryExpr(S);
        return mse(pred, Y);
    };

    float a = -4.f;
    float best_a = a;
    const float step = 0.01f;

    const int MAX_EPOCHS = 20;
    int epoch = 0;

    float best_cost = std::numeric_limits<float>::max();

    while (epoch++ < MAX_EPOCHS) {

        float cost_init = cost_calc(a, X, Y);

        if (cost_init < best_cost) {
            best_cost = cost_init;
            best_a = a;
        }

        // stopping if cost is too small
        if (cost_init < 1.f) break;

        a += step;
        float cost_end = cost_calc(a + step, X, Y);

        float grad = (cost_end - cost_init) / step;
        std::cout << "epoch:\t" << epoch << "\ta:\t" << a << "\tMSE:\t" << cost_init << "\tgrad:\t" << grad << "\n";

        // stopping if grad is too small
        if (abs(grad) < 10e-5) break;

        a = a - step * grad;

    }

    const float b = Cy - best_a*Cx;

    std::cout << "Best configuration is (a, b) = (" << best_a << ", " << b << ")" << " with cost " << best_cost << "\n";

    return 0;
}
