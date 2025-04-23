#include <iostream>
#include <exception>

#include "book/definitions.hpp"
#include "book/utils.hpp"

Eigen::Tensor<int, 2> calc_confusion_matrix(const Tensor_2D &TRUE, const Tensor_2D &PRED)
{
    if (TRUE.dimension(0) != PRED.dimension(0) || TRUE.dimension(1) != PRED.dimension(1))
    {
        throw std::invalid_argument("The dimensions of parameters do not match");
    }
    if (TRUE.dimension(1) < 2)
    {
        throw std::invalid_argument("The number of classes need to be greater or equals to 2");
    }

    const int num_registers = TRUE.dimension(0);
    const int num_classes = TRUE.dimension(1);
    Eigen::Tensor<int, 2> result(num_classes, num_classes);
    result.setZero();

    const Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(1, 0)};
    DimArray<1> max_dim({1});
    DimArray<1> sum_dim({0});
    DimArray<2> bcast_dim = {1, num_classes};
    const Eigen::array<int, 2> reshape_dim = {num_registers, 1};

    auto PRED_one_hot = book::utils::one_hot<TYPE>(PRED);

    for (size_t clazz = 0; clazz < num_classes; ++clazz)
    {

        Tensor_2D mold(num_classes, num_classes);
        mold.setZero();
        mold(clazz, clazz) = 1;

        auto matches = TRUE.contract(mold, contract_dims).maximum(max_dim).reshape(reshape_dim).broadcast(bcast_dim);

        auto prod = (PRED_one_hot * matches).cast<int>();
        auto conters = prod.sum(sum_dim);

        result.chip<0>(clazz) = conters;
    }

    return result;
}

float precision(const size_t class_index, const Eigen::Tensor<int, 2> & confusion_matrix) {
    auto column = confusion_matrix.chip<1>(class_index);
    float total = ((Eigen::Tensor<int, 0>)(column.sum()))(0);
    float tp = confusion_matrix(class_index, class_index);
    float result = tp / total;
    return result;
}

float recall(const size_t class_index, const Eigen::Tensor<int, 2> & confusion_matrix) {
    auto row = confusion_matrix.chip<0>(class_index);
    float total = ((Eigen::Tensor<int, 0>)(row.sum()))(0);
    float tp = confusion_matrix(class_index, class_index);
    float result = tp / total;
    return result;
}

float accuracy(const Eigen::Tensor<int, 2> & confusion_matrix) {
    Eigen::Tensor<int, 0> trace = confusion_matrix.trace();
    float diagonal = trace(0);
    float total = ((Eigen::Tensor<int, 0>)(confusion_matrix.sum()))(0);
    float result = diagonal / total;
    return result;
}

float f1_score(const size_t class_index, const Eigen::Tensor<int, 2> & confusion_matrix) {
    float _precision = precision(class_index, confusion_matrix);
    float _recall = recall(class_index, confusion_matrix);
    float result = 2*_precision*_recall/ (_precision + _recall);
    return result;
}

int main(int, char **)
{

    Tensor_2D TRUE(20, 3);
    TRUE.setValues({{0, 1, 0}, {1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {1, 0, 0}, {0, 1, 0}, {0, 1, 0}, {1, 0, 0},
                    {0, 1, 0}, {0, 1, 0}, {1, 0, 0}, {0, 0, 1}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0}, {0, 1, 0}, {1, 0, 0}, {0, 0, 1}});

    Tensor_2D PRED(20, 3);
    PRED.setValues({{1, 0, 0}, {1, 0, 0}, {0, 0, 1}, {1, 0, 0}, 
                        {0, 1, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 0},
                        {0, 1, 0}, {1, 0, 0},{0, 1, 0}, {1, 0, 0}, 
                        {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 1, 0}, 
                        {0, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}});

    // PRED.setValues({{0.44, 0.37, 0.18}, {0.61, 0.29, 0.09}, {0.01, 0.06, 0.92}, {0.67, 0.19, 0.13154}, 
    //                 {0.11, 0.83, 0.05}, {0.2, 0.30, 0.48}, {0.04, 0.60, 0.34}, {0.19, 0.77, 0.027}, 
    //                 {0.43, 0.54, 0.02}, {0.61, 0.12, 0.25}, {0.45, 0.47, 0.06}, {0.99, 0.0, 0.01}, 
    //                 {0.47, 0.44, 0.07}, {0.11, 0.75, 0.13}, {0.09, 0.28, 0.62}, {0.31, 0.36, 0.32}, 
    //                 {0.39, 0.56, 0.04}, {0.22, 0.53, 0.24}, {0.13, 0.08, 0.77}, {0.92, 0.08, 0.0}});

    auto confusion_matrix = calc_confusion_matrix(TRUE, PRED);

    std::cout << confusion_matrix << "\n";

    std::cout << "\nprecision class 0: " << precision(0, confusion_matrix) << "\n";
    std::cout << "precision class 1: " << precision(1, confusion_matrix) << "\n";
    std::cout << "precision class 2: " << precision(2, confusion_matrix) << "\n\n";

    std::cout << "recall class 0: " << recall(0, confusion_matrix) << "\n";
    std::cout << "recall class 1: " << recall(1, confusion_matrix) << "\n";
    std::cout << "recall class 2: " << recall(2, confusion_matrix) << "\n\n";

    std::cout << "f1_score class 0: " << f1_score(0, confusion_matrix) << "\n";
    std::cout << "f1_score class 1: " << f1_score(1, confusion_matrix) << "\n";
    std::cout << "f1_score class 2: " << f1_score(2, confusion_matrix) << "\n\n";

    std::cout << "accuracy: " << accuracy(confusion_matrix) << "\n";

    return 0;
}
