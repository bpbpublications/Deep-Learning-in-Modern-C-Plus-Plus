#include "book/metrics.hpp"
#include "book/utils.hpp"

TYPE intersection_over_union(const BoundingBox &A, const BoundingBox &B) {
    TYPE result = TYPE(0.);

    // calculating the intersection area
    TYPE xA = std::max(A.xmin, B.xmin);
    TYPE yA = std::max(A.ymin, B.ymin);
    TYPE xB = std::min(A.xmax, B.xmax);
    TYPE yB = std::min(A.ymax, B.ymax);
    TYPE intersection_area = std::max(TYPE(0.), xB - xA) * std::max(TYPE(0.), yB - yA);

    // Calculating union
    TYPE area_A = (A.xmax - A.xmin) * (A.ymax - A.ymin);
    TYPE area_B = (B.xmax - B.xmin) * (B.ymax - B.ymin);
    TYPE union_area = area_A + area_B - intersection_area;
    
    if (union_area > TYPE(0.)) {
        result = intersection_area / union_area;
    }

    return result;
}

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

        auto prod = PRED_one_hot * matches;
        auto conters = prod.sum(sum_dim).cast<int>();

        result.chip<0>(clazz) = conters;
    }

    return result;
}