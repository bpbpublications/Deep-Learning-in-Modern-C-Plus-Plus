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

template <typename Device, int INPUT_RANK, int OUTPUT_RANK>
class Layer
{
public:
    Layer(Device &device, std::string name) : device(device), name(std::move(name)) {}
    virtual ~Layer() {}

    virtual void forward(const Tensor<INPUT_RANK> &input) = 0;

    virtual void backward(const Tensor<OUTPUT_RANK> &upstream, bool propagate = true) = 0;

    virtual void update(const TYPE learning_rate, int epoch) {};

    virtual Tensor<OUTPUT_RANK> predict(const Tensor<INPUT_RANK> &input) const = 0;

    const Tensor<OUTPUT_RANK> &get_output() const
    {
        return this->output;
    }

    const Tensor<INPUT_RANK> &get_downstream() const
    {
        return this->downstream;
    }

    const std::string &get_name() const
    {
        return this->name;
    }

protected:
    Device &device;
    Tensor<OUTPUT_RANK> output;
    Tensor<INPUT_RANK> downstream;

private:
    std::string name;
};

template <typename Device>
class DenseLayer : public Layer<Device, 2, 2>
{
public:
    DenseLayer(Device &device, Tensor_2D weight, Tensor_1D bias, ActivationFunction<Device, 2> *activation_function) : Layer<Device, 2, 2>(device, "book.layers.dense"), 
        weight(std::move(weight)), bias(std::move(bias)), activation_function(activation_function)
    {

        if (this->weight.dimension(1) != this->bias.dimension(0))
            throw std::invalid_argument("bias and weight dimensions do not match.");

        this->weight_grad = this->weight.constant(0);
        this->bias_grad = this->bias.constant(0);
        this->use_bias = true;

    }

    DenseLayer(Device &device, Tensor_2D weight, bool use_bias, ActivationFunction<Device, 2> *activation_function) : Layer<Device, 2, 2>(device, "book.layers.dense"), 
        weight(std::move(weight)), use_bias(use_bias), activation_function(activation_function)
    {
        this->weight_grad = this->weight.constant(0);

        if (use_bias) {
            this->bias = Tensor_1D(this->weight.dimension(1));
            this->bias.setConstant(0);
            this->bias_grad = this->bias.constant(0);
        }

    }

    DenseLayer(Device &device, Tensor_2D weight, ActivationFunction<Device, 2> *activation_function) : DenseLayer(device, weight, false, activation_function) {}

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
            throw std::invalid_argument("input dimension mismatch: [" + std::to_string(input.dimension(0)) + ", "  + std::to_string(input.dimension(1)) + "] and " + std::to_string(this->weight.dimension(0)));
        }

        const Eigen::array<Eigen::IndexPair<TYPE>, 1> dims = {Eigen::IndexPair<TYPE>(1, 0)};
        const int batch_size = input.dimension(0);
        Tensor_2D Z = Tensor_2D(batch_size, this->weight.dimension(1));
        
        if (this->use_bias) {
            // Now, bias has the same dimensions as Z
            DimArray<2> dims_2D({1, this->bias.dimension(0)});
            DimArray<2> bcast({batch_size, 1});
            auto bias_reshaped = this->bias.reshape(dims_2D).broadcast(bcast);
            // This line does Z = W*X + bias
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
            throw std::invalid_argument("input dimension mismatch: [" + std::to_string(input.dimension(0)) + ", "  + std::to_string(input.dimension(1)) + "] and " + std::to_string(this->weight.dimension(0)));
        }

        this->last_input = input;
        const Eigen::array<Eigen::IndexPair<TYPE>, 1> dims = {Eigen::IndexPair<TYPE>(1, 0)};
        resize_tensor(this->activation, input.dimension(0), this->weight.dimension(1));
        
        this->activation.device(this->device) = input.contract(this->weight, dims);

        if (this->use_bias) {
            const int batch_size = this->activation.dimension(0);
            DimArray<2> dims_2D({1, this->bias.dimension(0)});
            DimArray<2> bcast({batch_size, 1});
            auto bias_reshaped = this->bias.reshape(dims_2D).broadcast(bcast);
            this->activation.device(this->device) = input.contract(this->weight, dims) + bias_reshaped;
        } else {
            this->activation.device(this->device) = input.contract(this->weight, dims);
        }

        resize_tensor(this->output, input.dimension(0), this->weight.dimension(1));
        this->activation_function->evaluate(this->activation, this->output);
    }

    virtual void backward(const Tensor_2D &upstream, bool propagate = true) {

        if (upstream.dimension(1) != this->weight.dimension(1)) {
            throw std::invalid_argument("upstream dimension mismatch: [" + std::to_string(upstream.dimension(0)) + ", "  + std::to_string(upstream.dimension(1)) + "].");
        }

        resize_tensor(this->dY_dZ, this->output.dimension(0), this->output.dimension(1));
        resize_tensor(this->dC_dZ, dY_dZ.dimension(0), dY_dZ.dimension(1));
        this->activation_function->derivative(this->output, this->dY_dZ);
        this->dC_dZ.device(this->device) = upstream * dY_dZ;
        this->calc_gradient(this->dC_dZ, propagate);
    }

    void update(const TYPE learning_rate, int epoch) {

        Tensor_2D weight_update = this->weight_grad * this->weight_grad.constant(-learning_rate);

        if (this->L1_weight_decay_lambda > 0) {
            const int batch_size = this->output.dimension(0);
            TYPE weight_decay = (learning_rate * this->L1_weight_decay_lambda / batch_size);
            auto signal_decay = this->weight.unaryExpr([](TYPE val){return book::utils::signal(val);});
            Tensor_2D decay = signal_decay * this->weight.constant(weight_decay) * this->weight;
            weight_update = weight_update - decay;
        }

        if (this->L2_weight_decay_lambda > 0) {
            const int batch_size = this->output.dimension(0);
            TYPE weight_decay = (learning_rate * this->L2_weight_decay_lambda / batch_size);
            Tensor_2D decay = this->weight.constant(weight_decay) * this->weight;
            weight_update = weight_update - decay;
        }

        this->weight = this->weight + weight_update;

        if (this->use_bias) {
            auto bias_update = this->bias_grad * this->bias_grad.constant(-learning_rate);
            this->bias = this->bias + bias_update;
        }
    }

    const Tensor_2D &get_weight() const {
        return this->weight;
    }

    const Tensor_1D &get_bias() const {
        return this->bias;
    }

    const Tensor_2D &get_weight_grad() const {
        return this->weight_grad;
    }

    const Tensor_1D &get_bias_grad() const {
        return this->bias_grad;
    }

    void set_L1_weight_decay_lambda(TYPE val) {
        this->L1_weight_decay_lambda = val;
    }

    void set_L2_weight_decay_lambda(TYPE val) {
        this->L2_weight_decay_lambda = val;
    }

protected:
    Tensor_2D weight;
    Tensor_2D weight_grad;

    ActivationFunction<Device, 2> *activation_function;

    Tensor_2D last_input; // the last input, required to compute the downsrteam gradient
    Tensor_2D activation; // activation is the result Z = W*X

    bool use_bias;
    Tensor_1D bias_grad;
    Tensor_1D bias;

    TYPE L1_weight_decay_lambda = 0;
    TYPE L2_weight_decay_lambda = 0;

    void calc_gradient(const Tensor_2D &dC_dZ, bool propagate) {
        const int batch_size = this->last_input.dimension(0);

        const Eigen::array<Eigen::IndexPair<TYPE>, 1> product_dims_0_0 = {Eigen::IndexPair<TYPE>(0, 0)};
        this->weight_grad.device(this->device) = this->last_input.contract(dC_dZ, product_dims_0_0);
        this->weight_grad = book::utils::clip_by_norm(this->weight_grad);

        if (this->use_bias) {
            DimArray<1> bias_reduc_dim({0});
            this->bias_grad.device(this->device) = dC_dZ.sum(bias_reduc_dim);
            this->bias_grad = book::utils::clip_by_norm(this->bias_grad);
        }

        if (propagate) {
            resize_tensor(this->downstream, dC_dZ.dimension(0), this->weight.dimension(0));
            const Eigen::array<Eigen::IndexPair<TYPE>, 1> product_dims_1_1 = {Eigen::IndexPair<TYPE>(1, 1)};
            this->downstream.device(this->device) = dC_dZ.contract(this->weight, product_dims_1_1);
        }
    }

private:
    // Pre-allocating the memory to avoid the cost of reallocating for every backward call
    Tensor_2D dC_dZ;
    Tensor_2D dY_dZ;
};

template <typename Device>
class SoftmaxCrossEntropyLayer : public DenseLayer<Device>
{
public:
    SoftmaxCrossEntropyLayer(Device &device, Tensor_2D weight) : DenseLayer<Device>(device, std::move(weight), new Softmax<Device>(device)) {}
    SoftmaxCrossEntropyLayer(Device &device, Tensor_2D weight, Tensor_1D bias) : DenseLayer<Device>(device, std::move(weight), std::move(bias), new Softmax<Device>(device)) {}
    SoftmaxCrossEntropyLayer(Device &device, Tensor_2D weight, bool use_bias) : DenseLayer<Device>(device, std::move(weight), use_bias, new Softmax<Device>(device)) {}
    virtual ~SoftmaxCrossEntropyLayer() {
        if (this->activation_function)
        {
            delete this->activation_function;
            this->activation_function = nullptr;
        }
    }

    virtual void backward(const Tensor_2D &TRUE, bool propagate) {
        Tensor_2D dC_dZ = this->output - TRUE;
        this->calc_gradient(dC_dZ, propagate);
    }
};

template<typename Generator, int _RANK>
void dropout_mask(Generator& gen, Eigen::Tensor<bool, _RANK> &tensor, TYPE drop_prob)
{

    std::uniform_real_distribution<TYPE> uniform_distro(0., 1.);
    tensor = tensor.unaryExpr([&uniform_distro, &gen, &drop_prob](TYPE) { 
                                TYPE val = uniform_distro(gen);
                                return val >= drop_prob; 
                            });

}

template <typename Device, int _RANK, typename GEN>
class DropoutLayer : public Layer<Device, _RANK, _RANK>
{
public:
    DropoutLayer(Device &device, TYPE drop_prob, GEN & gen) : Layer<Device, _RANK, _RANK>(device, "book.layers.dropout"), drop_prob(drop_prob), gen(gen) {}

    virtual ~DropoutLayer() {}

    virtual Tensor<_RANK> predict(const Tensor<_RANK> &input) const {
        return input;
    }

    virtual void forward(const Tensor<_RANK> &input) {

        if (this->mask.size() != input.size()) {
            this->mask = Eigen::Tensor<bool, _RANK>(input.dimensions());
        }
        if (this->output.size() != input.size()) {
            this->output = Tensor<_RANK>(input.dimensions());
        }
        dropout_mask(this->gen, this->mask, this->drop_prob);
        auto zero = this->output.constant(0);
        TYPE keep = TYPE(1.) - this->drop_prob;
        auto scale = this->output.constant(TYPE(1.) / keep);
        this->output.device(this->device) = this->mask.select(input, zero) * scale;
    }

    virtual void backward(const Tensor<_RANK> &upstream, bool propagate) {
        if (propagate) {
            if (this->downstream.size() != this->mask.size()) {
                this->downstream = Tensor<_RANK>(this->mask.dimensions());
            }
            auto zero = upstream.constant(0);
            TYPE keep = TYPE(1.) - this->drop_prob;
            auto scale = upstream.constant(TYPE(1.) / keep);
            this->downstream.device(this->device) = this->mask.select(upstream, zero) * scale;
        }
    }

private:
    Eigen::Tensor<bool, _RANK> mask;
    TYPE drop_prob;
    GEN gen;
};

#endif