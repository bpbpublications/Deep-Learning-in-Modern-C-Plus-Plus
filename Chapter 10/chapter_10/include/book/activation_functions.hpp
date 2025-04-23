/*
 * This file is part of Coding Deep Learning from Scratch Book, BPB PUBLICATIONS .
 *
 * Author: Luiz doleron <doleron@gmail.com>
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
 *
 * This Source Code Form is subject to the terms of the Mozilla
 * Public License v. 2.0. If a copy of the MPL was not distributed
 * with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef ACTIVATION_FUNCTIONS_H_
#define ACTIVATION_FUNCTIONS_H_

#include <exception>

template <typename Device, int _RANK>
class ActivationFunction
{
public:
    ActivationFunction(Device &device, std::string name) : device(device), name(std::move(name)) {}
    virtual ~ActivationFunction() {}

    virtual Tensor<_RANK> evaluate(const Tensor<_RANK> &Z) const = 0;

    virtual Tensor<_RANK> derivative(const Tensor<_RANK> &Z) const = 0;

    virtual Tensor<_RANK + 1> jacobian(const Tensor<_RANK> &Z) const = 0;

    virtual void evaluate(const Tensor<_RANK> &Z, Tensor<_RANK> &target) const = 0;

    virtual void derivative(const Tensor<_RANK> &Z, Tensor<_RANK> &target) const = 0;

    const std::string &get_name() const
    {
        return this->name;
    }

protected:
    Device &device;

private:
    std::string name;
};

template <typename Device, int _RANK>
class ReLU : public ActivationFunction<Device, _RANK>
{
public:
    ReLU(Device &device) : ActivationFunction<Device, _RANK>(device, "book.activations.relu") {}
    virtual ~ReLU() {}

    virtual Tensor<_RANK> evaluate(const Tensor<_RANK> &Z) const
    {
        auto zero = Z.constant(0.f);
        Tensor<_RANK> result = (Z > zero).select(Z, zero);
        return std::move(result);
    }

    virtual void evaluate(const Tensor<_RANK> &Z, Tensor<_RANK> &target) const
    {
        auto zero = Z.constant(0.f);
        target.device(this->device) = (Z > zero).select(Z, zero);
    }

    virtual Tensor<_RANK> derivative(const Tensor<_RANK> &Z) const
    {
        auto one = Z.constant(1.f);
        auto zero = Z.constant(0.f);
        Tensor<_RANK> result = (Z > zero).select(one, zero);
        return std::move(result);
    }

    virtual void derivative(const Tensor<_RANK> &Z, Tensor<_RANK> &target) const
    {
        auto one = Z.constant(1.f);
        auto zero = Z.constant(0.f);
        target.device(this->device) = (Z > zero).select(one, zero);
    }

    Eigen::Tensor<TYPE, 3> jacobian(const Tensor_2D &Z) const
    {

        Tensor_2D T = Z.unaryExpr([](TYPE v) {
                if (v > 0.f) return 1.f;
                return 0.f;
            }
        );

        const auto dimensions = T.dimensions();
        const int batch_length = dimensions[0];
        const int S = dimensions[1];

        const DimArray<3> reshape_dimensions = {batch_length, 1, S};

        const auto T_reshaped = T.reshape(reshape_dimensions);

        Tensor_2D _2D_diagonal(S, S);
        _2D_diagonal.setZero();
        for (int i = 0; i < S; ++i) _2D_diagonal(i, i) = 1.f;

        const DimArray<3> diagonal_dimensions = {1, S, S};

        const auto diagonal_reshaped = _2D_diagonal.reshape(diagonal_dimensions);

        const DimArray<3> diagonal_bcast = {batch_length, 1, 1};

        auto diagonal = diagonal_reshaped.broadcast(diagonal_bcast);

        const DimArray<3> T_bcast = {1, S, 1};

        auto T_extended = T_reshaped.broadcast(T_bcast);

        const Eigen::Tensor<TYPE, 3> result = T_extended * diagonal;

        return result;
    }

};

template <typename Device>
class Softmax : public ActivationFunction<Device, 2>
{
public:
    Softmax(Device &device) : ActivationFunction<Device, 2>(device, "book.activations.softmax") {}
    virtual ~Softmax() {}

    virtual Tensor_2D evaluate(const Tensor_2D &Z) const {
        Tensor_2D result(Z.dimension(0), Z.dimension(1));
        this->evaluate(Z, result);
        return result;
    }

    virtual void evaluate(const Tensor_2D &Z, Tensor_2D &target) const {
        auto dimensions = Z.dimensions();

        const int batch_length = dimensions[0];
        const int instance_length = dimensions[1];

        const Eigen::array<int, 2> reshape_dim = {batch_length, 1};
        const Eigen::array<int, 2> bcast = {1, instance_length};

        Eigen::array<int, 1> depth_dim({1});
        auto z_max = Z.maximum(depth_dim);
        auto max_reshaped = z_max.reshape(reshape_dim);
        auto max_values = max_reshaped.broadcast(bcast);

        auto diff = Z - max_values;

        auto expo = diff.exp();
        auto expo_sums = expo.sum(depth_dim);
        auto sums_reshaped = expo_sums.reshape(reshape_dim);
        auto sums = sums_reshaped.broadcast(bcast);
        target.device(this->device) = expo / sums;

    }

    Tensor<3> jacobian(const Tensor<2> &Z) const
    {

        Tensor<2> T = this->evaluate(Z);

        const auto dimensions = T.dimensions();
        const int S = dimensions[1];

        DimArray<3> reshape_dimensions;
        reshape_dimensions[0] = dimensions[0];
        reshape_dimensions[1] = 1;
        reshape_dimensions[2] = S;

        const auto T_reshaped = T.reshape(reshape_dimensions);

        Tensor<2> _2D_diagonal(S, S);
        _2D_diagonal.setZero();
        for (int i = 0; i < S; ++i) _2D_diagonal(i, i) = 1.f;

        DimArray<3> diagonal_dimensions;
        diagonal_dimensions[0] = 1;
        diagonal_dimensions[1] = S;
        diagonal_dimensions[2] = S;

        const auto diagonal_reshaped = _2D_diagonal.reshape(diagonal_dimensions);

        DimArray<3> diagonal_bcast;
        diagonal_bcast[0] = dimensions[0];
        diagonal_bcast[1] = 1;
        diagonal_bcast[2] = 1;

        Tensor<3> diagonal = diagonal_reshaped.broadcast(diagonal_bcast);

        DimArray<3> T_bcast;
        for (int i = 0; i <= 2; ++i) T_bcast[i] = 1;
        T_bcast[1] = S;

        Tensor<3> T_extended = T_reshaped.broadcast(T_bcast);

        DimArray<3> transposed_dim;
        transposed_dim[0] = 0;
        transposed_dim[2] = 1;
        transposed_dim[1] = 2;

        Tensor<3> T_extended_transposed = T_extended.shuffle(transposed_dim);

        const auto prod = T_extended * T_extended_transposed;

        const Tensor<3> result = T_extended * diagonal - prod;

        return result;
    }

    virtual Tensor_2D derivative(const Tensor_2D &Z) const {
        throw std::runtime_error("Only the jaccobian of Softmax is well-defined. The coefficient-wise derivative is not allowed.");
    }

    virtual void derivative(const Tensor_2D &Z, Tensor_2D &target) const
    {
        throw std::runtime_error("Only the jaccobian of Softmax is well-defined. The coefficient-wise derivative is not allowed.");
    }

};

#endif