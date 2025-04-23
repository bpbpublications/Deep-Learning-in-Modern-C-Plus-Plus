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

};

const auto sigmoid_1d = [] (TYPE z) {
    TYPE result;
    if (z >= 45.f)
        result = 1.f;
    else if (z <= -45.f)
        result = 0.f;
    else
        result = 1.f / (1.f + std::exp(-z));
    return result;
};

const auto sigmoid_1d_derivative = [] (TYPE z) {
    TYPE y = sigmoid_1d(z);
    return (1.f - y) * y;
};

template <typename Device, int _RANK>
class Sigmoid : public ActivationFunction<Device, _RANK>
{
public:
    Sigmoid(Device &device) : ActivationFunction<Device, _RANK>(device, "book.activations.sigmoid") {}
    virtual ~Sigmoid() {}

    virtual Tensor<_RANK> evaluate(const Tensor<_RANK> &Z) const
    {
        Tensor<_RANK> result = Z.constant(0.f);
        result.device(this->device) = Z.cwiseMax(result);
        return std::move(result);
    }

    virtual void evaluate(const Tensor<_RANK> &Z, Tensor<_RANK> &target) const
    {
        target.device(this->device) = Z.unaryExpr(sigmoid_1d);
    }

    virtual Tensor<_RANK> derivative(const Tensor<_RANK> &Z) const
    {
        Tensor<_RANK> result = Z.unaryExpr(sigmoid_1d_derivative);
        return std::move(result);
    }

    virtual void derivative(const Tensor<_RANK> &Z, Tensor<_RANK> &target) const
    {
        target.device(this->device) = Z.unaryExpr(sigmoid_1d_derivative);
    }

};

template <typename Device, int _RANK>
class Linear : public ActivationFunction<Device, _RANK>
{
public:
    Linear(Device &device) : ActivationFunction<Device, _RANK>(device, "book.activations.relu") {}
    virtual ~Linear() {}

    virtual Tensor<_RANK> evaluate(const Tensor<_RANK> &Z) const
    {
        return Z;
    }

    virtual void evaluate(const Tensor<_RANK> &Z, Tensor<_RANK> &target) const
    {
        if(&Z != &target) target = Z;
    }

    virtual Tensor<_RANK> derivative(const Tensor<_RANK> &Z) const
    {
        return Z.constant(1.f);
    }

    virtual void derivative(const Tensor<_RANK> &Z, Tensor<_RANK> &target) const
    {
        target.device(this->device) = Z.constant(1.f);
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

    virtual Tensor_2D derivative(const Tensor_2D &Z) const {
        throw std::runtime_error("Only jaccobian of Softmax is well-defined. The derivative is not allowed.");
    }

    virtual void derivative(const Tensor_2D &Z, Tensor_2D &target) const
    {
        throw std::runtime_error("Only jaccobian of Softmax is well-defined. The derivative is not allowed.");
    }

};

#endif