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

#ifndef ACTIVATION_FUNCTIONS_H_
#define ACTIVATION_FUNCTIONS_H_

template <int NumIndices_>
class ActivationFunction
{
public:
    ActivationFunction(std::string name) : name(std::move(name)) {}
    virtual ~ActivationFunction() {}

    virtual Tensor<NumIndices_> evaluate(const Tensor<NumIndices_> &Z) const = 0;

    virtual Tensor<NumIndices_ + 1> jacobian(const Tensor<NumIndices_> &Z) const = 0;

    virtual Tensor<NumIndices_> operator()(const Tensor<NumIndices_> &Z) const
    {
        return this->evaluate(Z);
    }

    const std::string &get_name() const
    {
        return this->name;
    }

private:
    std::string name;
};

template <int NumIndices_>
class Softmax : public ActivationFunction<NumIndices_>
{
public:
    Softmax() : ActivationFunction<NumIndices_>("book.activations.softmax") {}
    virtual ~Softmax() {}

    Tensor<NumIndices_> evaluate(const Tensor<NumIndices_> &Z) const
    {

        auto dimensions = Z.dimensions();

        int instance_length = dimensions.at(NumIndices_ - 1);

        Eigen::array<int, NumIndices_> reshape_dim;
        Eigen::array<int, NumIndices_> bcast;

        for (int i = 0; i < NumIndices_ - 1; ++i)
        {
            reshape_dim[i] = dimensions[i];
            bcast[i] = 1;
        }
        reshape_dim[NumIndices_ - 1] = 1;
        bcast[NumIndices_ - 1] = instance_length;

        Eigen::array<int, 1> depth_dim({NumIndices_ - 1});
        auto z_max = Z.maximum(depth_dim);
        auto max_reshaped = z_max.reshape(reshape_dim);
        auto max_values = max_reshaped.broadcast(bcast);

        // avoiding overflow
        auto diff = Z - max_values;

        auto expo = diff.exp();
        auto expo_sums = expo.sum(depth_dim);
        auto sums_reshaped = expo_sums.reshape(reshape_dim);
        auto sums = sums_reshaped.broadcast(bcast);
        Tensor<NumIndices_> result = expo / sums;

        return result;
    }

    Tensor<NumIndices_ + 1> jacobian(const Tensor<NumIndices_> &Z) const
    {

        Tensor<NumIndices_> T = this->evaluate(Z);

        const auto dimensions = T.dimensions();
        const int S = dimensions[NumIndices_ - 1];

        Eigen::array<Eigen::Index, NumIndices_ + 1> reshape_dimensions;
        for (int i = 0; i < NumIndices_ - 1; ++i)
        {
            reshape_dimensions[i] = dimensions[i];
        }

        reshape_dimensions[NumIndices_ - 1] = 1;
        reshape_dimensions[NumIndices_] = dimensions[NumIndices_ - 1];

        const auto T_reshaped = T.reshape(reshape_dimensions);

        Tensor<2> _2D_diagonal(S, S);
        _2D_diagonal.setZero();
        for (int i = 0; i < S; ++i)
            _2D_diagonal(i, i) = 1.f;

        Eigen::array<Eigen::Index, NumIndices_ + 1> diagonal_dimensions;

        for (int i = 0; i < NumIndices_ - 1; ++i)
        {
            diagonal_dimensions[i] = 1;
        }

        diagonal_dimensions[NumIndices_ - 1] = S;
        diagonal_dimensions[NumIndices_] = S;

        const auto diagonal_reshaped = _2D_diagonal.reshape(diagonal_dimensions);

        Eigen::array<Eigen::Index, NumIndices_ + 1> diagonal_bcast;
        for (int i = 0; i < NumIndices_ - 1; ++i)
        {
            diagonal_bcast[i] = dimensions[i];
        }

        diagonal_bcast[NumIndices_ - 1] = 1;
        diagonal_bcast[NumIndices_] = 1;

        Tensor<NumIndices_ + 1> diagonal = diagonal_reshaped.broadcast(diagonal_bcast);

        Eigen::array<Eigen::Index, NumIndices_ + 1> T_bcast;
        for (int i = 0; i <= NumIndices_; ++i)
        {
            T_bcast[i] = 1;
        }

        T_bcast[NumIndices_ - 1] = S;

        Tensor<NumIndices_ + 1> T_extended = T_reshaped.broadcast(T_bcast);

        Eigen::array<Eigen::Index, NumIndices_ + 1> transposed_dim;
        for (int i = 0; i <= NumIndices_; ++i)
        {
            transposed_dim[i] = i;
        }

        transposed_dim[NumIndices_] = NumIndices_ - 1;
        transposed_dim[NumIndices_ - 1] = NumIndices_;

        Tensor<NumIndices_ + 1> T_extended_transposed = T_extended.shuffle(transposed_dim);

        const auto prod = T_extended * T_extended_transposed;

        const Tensor<NumIndices_ + 1> result = T_extended * diagonal - prod;

        return result;
    }
};

#endif