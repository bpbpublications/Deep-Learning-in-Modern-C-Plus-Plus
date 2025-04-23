/*
 * This file is part of Coding Deep Learning from Scratch Book, BPB PUBLICATIONS .
 *
 * Author: Luiz doleron <doleron@gmail.com>
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * This Source Code Form is subject to the terms of the Mozilla
 * Public License v. 2.0. If a copy of the MPL was not distributed
 * with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef METRICS_H_
#define METRICS_H_

#include "book/definitions.hpp"
#include "book/metrics.hpp"

template <int _RANK>
TYPE accuracy(const Tensor<_RANK> &REAL, const Tensor<_RANK> &PRED)
{

    const auto compare = [](Eigen::DenseIndex a, Eigen::DenseIndex b) {
        return static_cast<TYPE>(a == b);
    };

    Eigen::Tensor<Eigen::DenseIndex, _RANK - 1> REAL_MAX = REAL.argmax(_RANK - 1);
    Eigen::Tensor<Eigen::DenseIndex, _RANK - 1> PRED_MAX = PRED.argmax(_RANK - 1);

    auto diff = REAL_MAX.binaryExpr(PRED_MAX, compare);

    Tensor_0D mean = diff.mean();

    TYPE result = mean(0) * TYPE(100.);

    return result;
}

template <int _RANK>
int count_success(const Tensor<_RANK> &REAL, const Tensor<_RANK> &PRED)
{

    const auto compare = [](Eigen::DenseIndex a, Eigen::DenseIndex b) {
        return static_cast<int>(a == b);
    };

    Eigen::Tensor<Eigen::DenseIndex, _RANK - 1> REAL_MAX = REAL.argmax(_RANK - 1);
    Eigen::Tensor<Eigen::DenseIndex, _RANK - 1> PRED_MAX = PRED.argmax(_RANK - 1);

    auto diff = REAL_MAX.binaryExpr(PRED_MAX, compare);

    Eigen::Tensor<int, 0> summup = diff.sum();

    int result = summup(0);

    return result;
}

struct BoundingBox {
    TYPE xmin;
    TYPE ymin;
    TYPE xmax;
    TYPE ymax;
};

/**
 * Implementation of IoU - Intersection over union
*/
TYPE intersection_over_union(const BoundingBox &A, const BoundingBox &B);

Eigen::Tensor<int, 2> calc_confusion_matrix(const Tensor_2D &TRUE, const Tensor_2D &PRED);

#endif