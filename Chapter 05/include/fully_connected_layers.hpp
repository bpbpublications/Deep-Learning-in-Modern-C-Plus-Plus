#ifndef __MY_FC_LAYERS__
#define __MY_FC_LAYERS__

#include <unsupported/Eigen/CXX11/Tensor>

template <int _RANK>
auto sigmoid_activation(const Tensor<_RANK> &Z) {
    auto sigmoid = [](TYPE z)
        {
            float result;
            if (z >= 45.)
                result = 1.;
            else if (z <= -45.)
                result = 0.;
            else
                result = 1. / (1. + exp(-z));
            return result;
        };

    auto result = Z.unaryExpr(sigmoid);
    return result;
}

class Dense
{
public:

    Dense(Tensor_2D _weights,
             Tensor_1D _bias,
             std::function<Tensor_2D(const Tensor_2D &)> _activation): 
             weights(std::move(_weights)),
             bias(std::move(_bias)),
             activation(_activation) {}
    virtual ~Dense() {}

    virtual Tensor<2> evaluate(const Tensor<2> &input)
    {

        const auto input_dims = input.dimensions();
        const auto weight_dims = this->weights.dimensions();

        const int instance_size = input_dims[1];
        const int output_size = weight_dims[1];

        // performing some runtime checks

        if (instance_size != weight_dims[0])
            throw std::invalid_argument("input size does not match the shape of the weight matrix");
            
        if (this->bias.dimension(0) != output_size)
            throw std::invalid_argument("bias size does not match the shape of the weight matrix");

        int instances = input.dimension(0);

        Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(1, 0)};
        Tensor_2D prod = input.contract(weights, contract_dims);

        // broadcasting the bias to match the output dimensions
        DimArray<2> bias_new_dim({1, output_size});
        auto bias_reshaped = this->bias.reshape(bias_new_dim);
        DimArray<2> bias_bcast({instances, 1});
        Tensor_2D bias_broadcast = bias_reshaped.broadcast(bias_bcast);

        // adding the bias
        auto Z = prod + bias_broadcast;

        // applying activation function
        auto result = this->activation(Z);

        return result;
    }

    virtual Tensor_2D 
    operator()(const Tensor_2D &input)
    {
        return this->evaluate(input);
    }

    int size()
    {
        return this->bias.size() + this->weights.size();
    }

private:
    Tensor_2D weights;
    Tensor_1D bias;
    std::function<Tensor_2D(const Tensor_2D &)> activation;
};

template <int _RANK>
auto flatten(const Tensor<_RANK> &input) {

    const auto input_dims = input.dimensions();

    const int batch_size = input_dims[0];
    int instance_size = input.size() / batch_size;

    DimArray<2> new_dim({batch_size, instance_size});
    Tensor_2D result = input.reshape(new_dim);

    return result;
}

#endif