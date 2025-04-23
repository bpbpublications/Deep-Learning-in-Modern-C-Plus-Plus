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

template <int _RANK>
using Optimizer = std::function<Tensor<_RANK>(Tensor<_RANK>&, TYPE, int)>;

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

        this->weight_optimizer = DEFAULT_UPDATE<2>();
        this->bias_optimizer = DEFAULT_UPDATE<1>();
    }

    DenseLayer(Device &device, Tensor_2D weight, bool use_bias, ActivationFunction<Device, 2> *activation_function) : Layer<Device, 2, 2>(device, "book.layers.dense"), 
        weight(std::move(weight)), use_bias(use_bias), activation_function(activation_function)
    {
        this->weight_grad = this->weight.constant(0);

        this->weight_optimizer = DEFAULT_UPDATE<2>();

        if (use_bias) {
            this->bias = Tensor_1D(this->weight.dimension(1));
            this->bias.setConstant(0);
            this->bias_grad = this->bias.constant(0);
            this->bias_optimizer = DEFAULT_UPDATE<1>();
        }

    }

    DenseLayer(Device &device, Tensor_2D weight, ActivationFunction<Device, 2> *activation_function) : DenseLayer(device, weight, false, activation_function) {}

    void set_weight_optimizer(Optimizer<2> opt) {
        this->weight_optimizer = opt;
    }

    void set_bias_optimizer(Optimizer<1> opt) {
        this->bias_optimizer = opt;
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
        auto weight_update = this->weight_optimizer(this->weight_grad, learning_rate, epoch);

        this->weight = this->weight + weight_update;

        if (this->use_bias) {
            auto bias_update = this->bias_optimizer(this->bias_grad, learning_rate, epoch);
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

protected:
    Tensor_2D weight;
    Tensor_2D weight_grad;

    ActivationFunction<Device, 2> *activation_function;

    Tensor_2D last_input; // the last input, required to compute the downsrteam gradient
    Tensor_2D activation; // activation is the result Z = W*X

    bool use_bias;
    Tensor_1D bias_grad;
    Tensor_1D bias;

    Optimizer<2> weight_optimizer;
    Optimizer<1> bias_optimizer;

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

template <typename Device>
class Conv2DLayer : public Layer<Device, 4, 4>
{
public:
    Conv2DLayer(Device &device, Tensor_4D kernels, Tensor_1D bias, ActivationFunction<Device, 4> *activation_function, std::vector<int> pad) : 
        Layer<Device, 4, 4>(device, "book.layers.conv2d"), kernels(std::move(kernels)), bias(std::move(bias)), activation_function(activation_function), pad(std::move(pad))
    {

        if (this->kernels.dimension(3) != this->bias.dimension(0))
            throw std::invalid_argument("bias and weight dimensions do not match.");

        this->kernels_grad = this->kernels.constant(0);
        this->bias_grad = this->bias.constant(0);
        this->use_bias = true;

        this->kernels_optimizer = DEFAULT_UPDATE<4>();
        this->bias_optimizer = DEFAULT_UPDATE<1>();
    }

    Conv2DLayer(Device &device, Tensor_4D kernels, bool use_bias, ActivationFunction<Device, 4> *activation_function, std::vector<int> pad) : 
        Layer<Device, 4, 4>(device, "book.layers.conv2d"), kernels(std::move(kernels)), use_bias(use_bias), activation_function(activation_function), pad(std::move(pad))
    {
        this->kernels_grad = this->kernels.constant(0);
        this->kernels_optimizer = DEFAULT_UPDATE<4>();

        if (use_bias) {
            this->bias = Tensor_1D(this->kernels.dimension(3));
            this->bias.setConstant(0);
            this->bias_grad = this->bias.constant(0);
            this->bias_optimizer = DEFAULT_UPDATE<1>();
        }
        
    }

    Conv2DLayer(Device &device, Tensor_4D kernels, ActivationFunction<Device, 4> *activation_function, std::vector<int> pad) : 
        Conv2DLayer(device, kernels, false, activation_function, pad) {}

    void set_kernels_optimizer(Optimizer<4> opt) {
        this->kernels_optimizer = opt;
    }
    
    void set_bias_optimizer(Optimizer<1> opt) {
        this->bias_optimizer = opt;
    }

    virtual ~Conv2DLayer()
    {
        if (this->activation_function)
        {
            delete this->activation_function;
            this->activation_function = nullptr;
        }
    }

    virtual void forward(const Tensor_4D &input)
    {
        
        // storing for further usage (backward)
        this->last_input = input;

        const int batch_size = input.dimension(0);
        const int input_dim1 = input.dimension(1);
        const int input_dim2 = input.dimension(2);
        const int input_channels = input.dimension(3);
        const int output_dim1 = input.dimension(1) - this->kernels.dimension(0) + 1 + this->pad[0] + this->pad[1];
        const int output_dim2 = input.dimension(2) - this->kernels.dimension(1) + 1 + this->pad[2] + this->pad[3];
        const int filters = this->kernels.dimension(3);

        // pre allocating memory
        resize_tensor(this->activation, batch_size, output_dim1, output_dim2, filters);
        resize_tensor(this->output, batch_size, output_dim1, output_dim2, filters);

        // computing the pad only once
        Eigen::array<std::pair<int, int>, 4> padding;
        padding[0] = std::make_pair(0, 0);
        padding[1] = std::make_pair(this->pad[0], this->pad[1]);
        padding[2] = std::make_pair(this->pad[2], this->pad[3]);
        padding[3] = std::make_pair(0, 0);
        Tensor_4D padded(batch_size, input_dim1 + this->pad[0] + this->pad[1], input_dim2 + this->pad[2] + this->pad[3], input_channels);
        padded.device(this->device) = input.pad(padding);

        // computing each channel
        const Eigen::array<Eigen::Index, 3> conv_dims({1, 2, 3});
        const Eigen::array<Eigen::Index, 3> chip_dims{{batch_size, output_dim1, output_dim2}};

        std::vector<int> indexes(filters);
        std::iota(indexes.begin(), indexes.end(), 0);

        const auto op = [&](int k)
        {
            Tensor_3D kernel = this->kernels.chip<3>(k);
            this->activation.chip<3>(k).device(this->device) = padded.convolve(kernel, conv_dims).reshape(chip_dims);

            if (this->use_bias) {
                this->activation.chip<3>(k).device(this->device) = padded.convolve(kernel, conv_dims).reshape(chip_dims) + this->bias[k];
            } else {
                this->activation.chip<3>(k).device(this->device) = padded.convolve(kernel, conv_dims).reshape(chip_dims);
            }
        };

        std::for_each(std::execution::par_unseq, indexes.begin(), indexes.end(), op);

        // applying the activation function inplace
        this->activation_function->evaluate(this->activation, this->output);
    }

    virtual void backward(const Tensor_4D &upstream, bool propagate = true)
    {

        resize_tensor(this->dY_dZ, this->activation.dimension(0), this->activation.dimension(1), this->activation.dimension(2), this->activation.dimension(3));
        resize_tensor(this->dC_dZ, this->activation.dimension(0), this->activation.dimension(1), this->activation.dimension(2), this->activation.dimension(3));
        // getting dY/dZ
        this->activation_function->derivative(this->activation, this->dY_dZ);
        // getting dC/dZ
        this->dC_dZ.device(this->device) = upstream * dY_dZ;

        // padding the last input
        Eigen::array<std::pair<int, int>, 4> padding;
        padding[0] = std::make_pair(0, 0);
        padding[1] = std::make_pair(this->pad[0], this->pad[1]);
        padding[2] = std::make_pair(this->pad[2], this->pad[3]);
        padding[3] = std::make_pair(0, 0);
        // here we have a memory x time tradeoff:
        // using 'Tensor_4D padded' we actually allocate the whole padded input, but just once
        // using 'auto padded' we only get an operation which is recomputed k times later on in the loop

        const long int batch_size = this->last_input.dimension(0);
        const long int input_channels = this->last_input.dimension(3);

        resize_tensor(this->padded, batch_size, this->last_input.dimension(1) + this->pad[0] + this->pad[1],
                                       this->last_input.dimension(2) + this->pad[2] + this->pad[3], input_channels);
        this->padded.device(this->device) = this->last_input.pad(padding);

        Eigen::array<Eigen::Index, 3> conv_dims({0, 1, 2});
        Eigen::array<Eigen::Index, 3> chip_dims{{this->kernels.dimension(0), this->kernels.dimension(1), this->kernels.dimension(2)}};
        const long int filters = this->kernels.dimension(3);

        std::vector<int> indexes(filters);
        std::iota(indexes.begin(), indexes.end(), 0);

        DimArray<1> bias_reduc_dim({0});
        const auto op = [&](int k)
        {
            auto dC_dZ_k = dC_dZ.chip<3>(k);
            this->kernels_grad.chip<3>(k).device(this->device) = this->padded.convolve(dC_dZ_k, conv_dims).reshape(chip_dims);

            if (this->use_bias) {
                Tensor_0D dC_dZ_k_sum = dC_dZ_k.sum();
                this->bias_grad(k) = dC_dZ_k_sum(0);
            }
        };

        std::for_each(std::execution::par_unseq, indexes.begin(), indexes.end(), op);

        if (propagate) {

            Eigen::array<bool, 4> reverse({true, true, false, false});
            Tensor_4D kernels_180 = this->kernels.reverse(reverse);

            const int p_h = (this->last_input.dimension(1) - dC_dZ.dimension(1) + kernels_180.dimension(0) - 1) / 2;
            const int p_w = (this->last_input.dimension(2) - dC_dZ.dimension(2) + kernels_180.dimension(1) - 1) / 2;
            Eigen::array<std::pair<int, int>, 4> dC_dZ_padding;
            dC_dZ_padding[0] = std::make_pair(0, 0);
            dC_dZ_padding[1] = std::make_pair(p_h, p_h);
            dC_dZ_padding[2] = std::make_pair(p_w, p_w);
            dC_dZ_padding[3] = std::make_pair(0, 0);

            resize_tensor(this->dC_dZ_padded, batch_size, dC_dZ.dimension(1) + 2*p_h, dC_dZ.dimension(2) + 2*p_w, filters);
            this->dC_dZ_padded.device(this->device) = dC_dZ.pad(dC_dZ_padding);

            std::vector<int> channels(this->last_input.dimension(3));
            std::iota(channels.begin(), channels.end(), 0);
            
            Eigen::array<Eigen::Index, 3> propagate_dims({1, 2, 3});
            resize_tensor(this->downstream, batch_size, this->last_input.dimension(1), this->last_input.dimension(2), input_channels);
            Tensor_4D &_downstream = this->downstream;

            const Eigen::array<Eigen::Index, 3> propag_dims{{batch_size, this->last_input.dimension(1), this->last_input.dimension(2)}};
            const auto propagate_op = [&](int c)
            {
                auto kernel = kernels_180.chip<2>(c);
                _downstream.chip<3>(c).device(this->device) = this->dC_dZ_padded.convolve(kernel, propagate_dims).reshape(propag_dims);
            };

            std::for_each(std::execution::par_unseq, channels.begin(), channels.end(), propagate_op);
        }
    }

    virtual Tensor_4D predict(const Tensor_4D &input) const
    {

        const int batch_size = input.dimension(0);
        const int input_dim1 = input.dimension(1);
        const int input_dim2 = input.dimension(2);
        const int input_channels = input.dimension(3);
        const int output_dim1 = input_dim1 - this->kernels.dimension(0) + 1 + this->pad[0] + this->pad[1];
        const int output_dim2 = input_dim2 - this->kernels.dimension(1) + 1 + this->pad[2] + this->pad[3];
        const int filters = this->kernels.dimension(3);

        Eigen::array<std::pair<int, int>, 4> padding;
        padding[0] = std::make_pair(0, 0);
        padding[1] = std::make_pair(this->pad[0], this->pad[1]);
        padding[2] = std::make_pair(this->pad[2], this->pad[3]);
        padding[3] = std::make_pair(0, 0);

        Tensor_4D result(batch_size, output_dim1, output_dim2, filters);

        Tensor_4D padded(batch_size, input_dim1 + this->pad[0] + this->pad[1], input_dim2 + this->pad[2] + this->pad[3], input_channels);
        padded.device(this->device) = input.pad(padding);

        const Eigen::array<Eigen::Index, 3> conv_dims({1, 2, 3});
        const Eigen::array<Eigen::Index, 3> chip_dims{{batch_size, output_dim1, output_dim2}};

        std::vector<int> indexes(filters);
        std::iota(indexes.begin(), indexes.end(), 0);

        const auto op = [&](int k)
        {
            auto kernel = this->kernels.chip<3>(k);
            if (this->use_bias) {
                result.chip<3>(k).device(this->device) = padded.convolve(kernel, conv_dims).reshape(chip_dims) + this->bias[k];
            } else {
                result.chip<3>(k).device(this->device) = padded.convolve(kernel, conv_dims).reshape(chip_dims);
            }
        };

        std::for_each(std::execution::par_unseq, indexes.begin(), indexes.end(), op);

        this->activation_function->evaluate(result, result);
        return std::move(result);
    }

    void update(const TYPE learning_rate, int epoch)
    {
        auto update = this->kernels_optimizer(this->kernels_grad, learning_rate, epoch);

        this->kernels = this->kernels + update;

        if (this->use_bias) {
            auto bias_update = this->bias_optimizer(this->bias_grad, learning_rate, epoch);
            this->bias = this->bias + bias_update;
        }
    }

    // for test purposes | testability 

    const Tensor_4D &get_kernels_grad() const
    {
        return this->kernels_grad;
    }

    const Tensor_1D &get_bias_grad() const
    {
        return this->bias_grad;
    }

    const Tensor_4D &get_kernels() const
    {
        return this->kernels;
    }

    const Tensor_1D &get_bias() const
    {
        return this->bias;
    }

protected:
    Tensor_4D kernels;
    Tensor_4D kernels_grad;

    bool use_bias;
    Tensor_1D bias;
    Tensor_1D bias_grad;

    ActivationFunction<Device, 4> *activation_function;

    Tensor_4D last_input;
    std::vector<int> pad;
    Tensor_4D activation;

    Optimizer<4> kernels_optimizer;
    Optimizer<1> bias_optimizer;

private:
    // pre-loading memory to avoid realloacte memory to each backward() call
    Tensor_4D dY_dZ;
    Tensor_4D dC_dZ;
    Tensor_4D padded;
    Tensor_4D dC_dZ_padded;
};

template <typename Device, int INPUT_RANK>
class FlattenLayer : public Layer<Device, INPUT_RANK, 2>
{
public:
    FlattenLayer(Device &device) : Layer<Device, INPUT_RANK, 2>(device, "book.layers.flatten") {
        suffle_dims[0] = 0;
        for (int i = 1; i < INPUT_RANK; ++i) {
            suffle_dims[i] = INPUT_RANK - i;
        }
    }

    virtual ~FlattenLayer() {}

    virtual Tensor_2D predict(const Tensor<INPUT_RANK> &input) const {
        auto transposed = input.shuffle(this->suffle_dims);
        const int batch_size = input.dimension(0);
        Eigen::array<Eigen::Index, 2> output_dims {batch_size, input.size() / batch_size};
        return transposed.reshape(output_dims);
    }

    virtual void forward(const Tensor<INPUT_RANK> &input) {
        auto transposed = input.shuffle(this->suffle_dims);
        const int batch_size = input.dimension(0);
        const int flat_dim = input.size() / batch_size;
        Eigen::array<Eigen::Index, 2> output_dims {batch_size, flat_dim};
        resize_tensor(this->output, batch_size, flat_dim);
        // Don't actually need device here
        this->output.device(this->device) = transposed.reshape(output_dims);

        this->input_dims[0] = batch_size;
        for (int i = 1; i < INPUT_RANK; ++i) {
            this->input_dims[i] = input.dimension(INPUT_RANK - i);
        }
        if (this->downstream.size() != input.size()) {
            this->downstream = input.constant(0);
        }
    }

    virtual void backward(const Tensor_2D &upstream, bool propagate) {
        if (propagate) {
            auto reshaped = upstream.reshape(input_dims);
            this->downstream.device(this->device) = reshaped.shuffle(this->suffle_dims);
        }
    }

private:
    Eigen::array<Eigen::Index, INPUT_RANK> suffle_dims;
    Eigen::array<Eigen::Index, INPUT_RANK> input_dims;
};

template <typename Device>
class MaxPooling : public Layer<Device, 4, 4>
{
public:
    MaxPooling(Device &device, int pool_size) : Layer<Device, 4, 4>(device, "book.layers.maxpooling"), pool_size(pool_size) {
        if (pool_size <= 1) throw std::invalid_argument("pool_size must be >= 2");
    }

    virtual ~MaxPooling() {}

    virtual void forward(const Tensor_4D &input) {

        const int batch_size = input.dimension(0);
        const int height = static_cast<int>(std::floor(static_cast<TYPE>(input.dimension(1)) / this->pool_size)) * this->pool_size;
        const int width = static_cast<int>(std::floor(static_cast<TYPE>(input.dimension(2)) / this->pool_size)) * this->pool_size;
        const int channels = input.dimension(3);

        // prunning leading columns and rows
        const Eigen::array<Eigen::Index, 4> offsets = {0, 0, 0, 0};
        const Eigen::array<Eigen::Index, 4> extents = {batch_size, height, width, channels};
        auto actual = input.slice(offsets, extents);

        // getting the patches
        auto patches = actual.extract_image_patches(this->pool_size, this->pool_size, this->pool_size, this->pool_size, Eigen::PADDING_VALID);

        // reducing to get the max of each patch
        const Eigen::array<Eigen::Index, 2> dims({1, 2});
        auto max_patches = patches.maximum(dims);

        // reshaping
        const int strides = pool_size;
        const int pre_rows = height / strides;
        const int pre_cols = width / strides;
        const Eigen::array<Eigen::Index, 4> pre_dims{{batch_size, pre_rows, pre_cols, channels}};
        resize_tensor(this->output, batch_size, pre_rows, pre_cols, channels);
        this->output.device(this->device) = max_patches.reshape(pre_dims);

        // preparing backward
        int pool_area = this->pool_size * this->pool_size;
        const Eigen::array<Eigen::Index, 4> argmax_dims{{batch_size, pool_area, height * width / pool_area, channels}};
        resize_tensor(this->mask, batch_size, pre_rows, pre_cols, channels);
        // storing max indexes
        this->mask.device(this->device) = patches.reshape(argmax_dims).argmax(1).reshape(pre_dims);
        resize_tensor(this->downstream, input.dimension(0), input.dimension(1), input.dimension(2), input.dimension(3));
    }

    // for debug purposes
    const Eigen::Tensor<Eigen::DenseIndex, 4> &get_mask() const
    {
        return this->mask;
    }

    /**
     * backproping maxpooling
     * 
     * the max element indexes were stored during forward and here they are used to propagate the upstream 
     * to downstream
     * 
     * for example, taking upstream:
     * 
     * [
     *    [[[2] [5]]]
     *    [[[8] [4]]]
     * ]
     * 
     * and mask:
     * 
     * [
     *   [[[1] [3]]]
     *   [[[2] [0]]]
     * ]
     * 
     * results in:
     * 
     * [
     *   [[[0] [0] [0] [0]]
     *    [[2] [0] [0] [5]]]
     *   [[[0] [8] [4] [0]]
     *    [[0] [0] [0] [0]]]
     * ]
    */
    virtual void backward(const Tensor_4D &upstream, bool propagate) {

        if (propagate) {

            this->downstream.setConstant(0);

            const size_t channels = this->mask.dimension(3);
            const size_t cols = this->mask.dimension(2);
            const size_t rows = this->mask.dimension(1);
            const size_t batch_size = this->mask.dimension(0);

            const int plane = cols * rows;
            const int planes = batch_size * cols * rows;

            const int size = this->mask.size();

            // transversing the mask tensor in parallel

            int threads = std::thread::hardware_concurrency();
            // reducing context change overhead
            if (size <= 4 * threads) {
                threads = 1;
            }
            const int par_group_size = std::ceil(static_cast<float>(size) / threads);
            std::vector<int> indexes(threads);
            std::iota(indexes.begin(), indexes.end(), 0);

            const auto op = [&](int k)
            {

                const int begin = k * par_group_size;
                const int limit = std::min(size, begin + par_group_size);
                for (int i = begin; i < limit; ++i) {

                    int instance = i % batch_size;
                    int channel = i / planes;
                    int row = (i / batch_size) % rows;
                    int col = (i / (batch_size * rows)) % cols;

                    Eigen::Index pos = this->mask(i);

                    int add_r = pos % this->pool_size;
                    int add_c = pos / this->pool_size;

                    int target_r = row * this->pool_size + add_r;
                    int target_c = col * this->pool_size + add_c;

                    this->downstream(instance, target_r, target_c, channel) = upstream(i);

                }
            };

            std::for_each(std::execution::par_unseq, indexes.begin(), indexes.end(), op);

        }

    }

    virtual Tensor_4D predict(const Tensor_4D &input) const {
        const int batch_size = input.dimension(0);
        const int height = static_cast<int>(std::floor(static_cast<TYPE>(input.dimension(1)) / this->pool_size)) * this->pool_size;
        const int width = static_cast<int>(std::floor(static_cast<TYPE>(input.dimension(2)) / this->pool_size)) * this->pool_size;
        const int channels = input.dimension(3);

        const Eigen::array<Eigen::Index, 4> offsets = {0, 0, 0, 0};
        const Eigen::array<Eigen::Index, 4> extents = {batch_size, height, width, channels};

        auto actual = input.slice(offsets, extents);

        auto patches = actual.extract_image_patches(this->pool_size, this->pool_size, this->pool_size, this->pool_size, Eigen::PADDING_VALID);

        Eigen::array<Eigen::Index, 2> dims({1, 2});
        auto max_patches = patches.maximum(dims);

        const int strides = pool_size;

        const int pre_rows = height / strides;
        const int pre_cols = width / strides;
        const Eigen::array<Eigen::Index, 4> pre_dims{{batch_size, pre_rows, pre_cols, channels}};

        Tensor_4D result(batch_size, pre_rows, pre_cols, channels);

        result.device(this->device) = max_patches.reshape(pre_dims);

        return std::move(result);
    }

private:
    int pool_size;
    Eigen::Tensor<Eigen::DenseIndex, 4> mask;

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

/**
 * A reshaping layer to convert 4D input tensors into 2D & vice-versa.
*/
template <typename Device, int INPUT_RANK, int OUTPUT_RANK>
class ReshapeLayer : public Layer<Device, INPUT_RANK, OUTPUT_RANK>
{
public:
    ReshapeLayer(Device &device, std::vector<Eigen::Index> target_dims) : Layer<Device, INPUT_RANK, OUTPUT_RANK>(device, "book.layers.reshape"), target_dims(std::move(target_dims)) {
        if (this->target_dims.size() != (OUTPUT_RANK - 1)) throw std::invalid_argument("dims must have " + std::to_string(OUTPUT_RANK - 1) + " elements.");
    }

    virtual ~ReshapeLayer() {}

    virtual Tensor<OUTPUT_RANK> predict(const Tensor<INPUT_RANK> &input) const {
        const int batch_size = input.dimension(0);
        Eigen::array<Eigen::Index, OUTPUT_RANK> output_dims;
        output_dims[0] = batch_size;
        for (int d = 0; d < target_dims.size(); ++d) {
            output_dims[d + 1] = target_dims[d];
        }
        return input.reshape(output_dims);
    }

    virtual void forward(const Tensor<INPUT_RANK> &input) {
        const int batch_size = input.dimension(0);
        Eigen::array<Eigen::Index, OUTPUT_RANK> output_dims;
        output_dims[0] = batch_size;
        for (int d = 0; d < target_dims.size(); ++d) {
            output_dims[d + 1] = target_dims[d];
        }
        this->output = input.reshape(output_dims);

        for (int d = 0; d < INPUT_RANK; ++d) {
            this->input_dims[d] = input.dimension(d);
        }
    }

    virtual void backward(const Tensor<OUTPUT_RANK> &upstream, bool propagate) {
        if (propagate) {
            this->downstream = upstream.reshape(input_dims);
        }
    }

private:
    std::vector<Eigen::Index> target_dims;
    Eigen::array<Eigen::Index, INPUT_RANK> input_dims;
};

#endif