/*of
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

#ifndef LAYERS_H_
#define LAYERS_H_

#include <execution>

#include <unsupported/Eigen/CXX11/Tensor>

/**
 * This function resizes the tensor using the indexes
 */
template <typename T, int _RANK, typename... IndexTypes>
void resize_tensor(Eigen::Tensor<T, _RANK> &tensor, IndexTypes... indexes)
{

    static_assert(sizeof...(indexes) == _RANK, "Tensor RANK and number of indexes do not match");

    long int size = 1;

    for (auto p : {indexes...})
    {
        size *= p;
    }

    bool resize = tensor.size() != size;

    if (resize)
    {
        tensor = Eigen::Tensor<T, _RANK>(indexes...);
    }
}

template <typename Device>
class DenseLayer
{
public:
    DenseLayer(Device &device, Tensor_2D weight, bool use_bias, ActivationFunction<Device, 2> *activation_function) : 
        device(device), weight(std::move(weight)), use_bias(use_bias), activation_function(activation_function)
    {
        this->weight_grad = this->weight.constant(0);

        if (use_bias) {
            this->bias = Tensor_1D(this->weight.dimension(1));
            this->bias.setConstant(0);
            this->bias_grad = this->bias.constant(0);
        }
    }

    virtual ~DenseLayer()
    {
        if (this->activation_function) {
            delete this->activation_function;
            this->activation_function = nullptr;
        }
    }

    /**
     * this method is designed to calculate the output of the layer during the prediction phase.
     * Thus, it is not suitable to be used for training the layer.
    */
    virtual Tensor_2D predict(const Tensor_2D &input) const
    {
        if (input.dimension(1) != this->weight.dimension(0)) {
            throw std::invalid_argument("input dimension mismatch: [" + std::to_string(input.dimension(0)) + ", "  
                + std::to_string(input.dimension(1)) + "] and " + std::to_string(this->weight.dimension(0)));
        }

        const Eigen::array<Eigen::IndexPair<TYPE>, 1> dims = {Eigen::IndexPair<TYPE>(1, 0)};
        const int batch_size = input.dimension(0);
        Tensor_2D Z = Tensor_2D(batch_size, this->weight.dimension(1));
        
        if (this->use_bias) {
            const int batch_size = this->activation.dimension(0);
            DimArray<2> dims_2D({1, this->bias.dimension(0)});
            DimArray<2> bcast({batch_size, 1});
            auto bias_reshaped = this->bias.reshape(dims_2D).broadcast(bcast);
            Z.device(this->device) = input.contract(this->weight, dims) + bias_reshaped;
        } else {
            Z.device(this->device) = input.contract(this->weight, dims);
        }

        Tensor_2D result = Tensor_2D(Z.dimension(0), Z.dimension(1));
        this->activation_function->evaluate(Z, result);
        return std::move(result);
    }

    /**
     * forward() works akin predict(). The difference here is that fowrad is design to the training phase.
     * Thus, unlike predict(), forward() stores computations (such as last_input, activation, and output) to be used
     * later, during the backward() call.
    */
    virtual void forward(const Tensor_2D &input)
    {
        if (input.dimension(1) != this->weight.dimension(0)) {
            throw std::invalid_argument("input dimension mismatch: [" + std::to_string(input.dimension(0)) + ", "  
                + std::to_string(input.dimension(1)) + "] and " + std::to_string(this->weight.dimension(0)));
        }

        this->last_input = input;
        const Eigen::array<Eigen::IndexPair<TYPE>, 1> dims = {Eigen::IndexPair<TYPE>(1, 0)};
        resize_tensor(this->activation, input.dimension(0), this->weight.dimension(1));
        
        if (this->use_bias) {
            const int batch_size = this->activation.dimension(0);
            DimArray<2> dims_2D({1, this->bias.dimension(0)});
            DimArray<2> bcast({batch_size, 1});
            auto bias_2d = this->bias.reshape(dims_2D).broadcast(bcast);
            this->activation.device(this->device) = input.contract(this->weight, dims) + bias_2d;
        } else {
            this->activation.device(this->device) = input.contract(this->weight, dims);
        }

        resize_tensor(this->output, input.dimension(0), this->weight.dimension(1));
        this->activation_function->evaluate(this->activation, this->output);
    }

    virtual void backward(const Tensor_2D &upstream, bool propagate = true) {

        if (upstream.dimension(1) != this->weight.dimension(1)) {
            throw std::invalid_argument("upstream dimension mismatch: [" + std::to_string(upstream.dimension(0)) + ", "  
                + std::to_string(upstream.dimension(1)) + "].");
        }

        resize_tensor(this->dY_dZ, this->output.dimension(0), this->output.dimension(1));
        resize_tensor(this->dC_dZ, dY_dZ.dimension(0), dY_dZ.dimension(1));
        this->activation_function->derivative(this->output, this->dY_dZ);
        this->dC_dZ.device(this->device) = upstream * dY_dZ;
        this->gradient(this->dC_dZ, propagate);
    }

    void update(const TYPE learning_rate, int epoch) {

        Tensor_2D weight_update = this->weight_grad * this->weight_grad.constant(-learning_rate);

        this->weight = this->weight + weight_update;

        if (this->use_bias) {
            Tensor_1D bias_update = this->bias_grad * this->bias_grad.constant(-learning_rate);
            this->bias = this->bias + bias_update;
        }

    }

    const Tensor_2D &get_output() const
    {
        return this->output;
    }

    const Tensor_2D &get_downstream() const
    {
        return this->downstream;
    }

protected:

    ActivationFunction<Device, 2> *activation_function;

    void gradient(const Tensor_2D &dC_dZ, bool propagate) {

        const Eigen::array<Eigen::IndexPair<TYPE>, 1> product_dims_0_0 = {Eigen::IndexPair<TYPE>(0, 0)};
        this->weight_grad.device(this->device) = this->last_input.contract(dC_dZ, product_dims_0_0);

        if (this->use_bias) {
            DimArray<1> bias_reduc_dim({0});
            this->bias_grad.device(this->device) = dC_dZ.sum(bias_reduc_dim);
        }

        if (propagate) {
            resize_tensor(this->downstream, dC_dZ.dimension(0), this->weight.dimension(0));
            const Eigen::array<Eigen::IndexPair<TYPE>, 1> product_dims_1_1 = {Eigen::IndexPair<TYPE>(1, 1)};
            this->downstream.device(this->device) = dC_dZ.contract(this->weight, product_dims_1_1);
        }
    }
    
    Tensor_2D output;
    Tensor_2D downstream;

private:
    Device &device;

    Tensor_2D weight;
    Tensor_2D weight_grad;

    bool use_bias;
    Tensor_1D bias;
    Tensor_1D bias_grad;
    
    Tensor_2D last_input; // the last input, required to compute the downsrteam gradient
    Tensor_2D activation; // activation is the result Z = W*X

    // Pre-allocating the memory to avoid the cost of reallocating for every backward call
    Tensor_2D dC_dZ;
    Tensor_2D dY_dZ;
};

template <typename Device>
class SoftmaxCrossEntropyLayer : public DenseLayer<Device>
{
public:
    SoftmaxCrossEntropyLayer(Device &device, Tensor_2D weight, bool use_bias) : DenseLayer<Device>(device, std::move(weight), use_bias, new Softmax<Device>(device)) {}

    virtual ~SoftmaxCrossEntropyLayer() {
        if (this->activation_function)
        {
            delete this->activation_function;
            this->activation_function = nullptr;
        }
    }

    virtual void backward(const Tensor_2D &TRUE, bool propagate) {
        const int batch_size = TRUE.dimension(0);
        Tensor_2D dC_dZ = (this->output - TRUE) / TRUE.constant(batch_size);
        this->gradient(dC_dZ, propagate);
    }
};

#endif