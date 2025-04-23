#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

#include "book/definitions.hpp"

template <int _RANK>
TYPE bce(const Tensor<_RANK> &PRED, const Tensor<_RANK> &TRUE) {
    auto COMP_TRUE = TRUE.constant(1.) - TRUE;
    auto COMP_PRED = PRED.constant(1.) - PRED;
    auto part1 = TRUE * (PRED + PRED.constant(1e-7)).log();
    auto part2 = COMP_TRUE * (COMP_PRED + COMP_PRED.constant(1e-7)).log();
    auto parts = part1 + part2;
    float sum = ((Tensor_0D)(parts.sum()))(0);
    float result = -sum / PRED.size();
    return result;
}

int main(int, char**)
{

    Tensor_2D TRUE(1, 10);
    TRUE.setValues({
        {1.f, 0.f, 1.f, 1.f, 0.f, 1.f, 1.f, 0.f, 1.f, 0.f}
    });
    Tensor_2D PRED(1, 10);
    PRED.setValues({
        {.59f, 0.12f, .9f, .77f, 0.f, .95f, 1.f, 0.22f, .55f, 0.123f}
    });

    std::cout << "TRUE: \n\n" << TRUE << "\n\n";
    std::cout << "PRED: \n\n" << PRED << "\n\n";

    std::cout << "BCE: \n\n" << bce(PRED, TRUE) << "\n\n";

    return 0;
}
