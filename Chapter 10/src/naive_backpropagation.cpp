#include <iostream>
#include <chrono>
using namespace std::chrono;

#include <unsupported/Eigen/CXX11/Tensor>

#include "book/book.hpp"

std::random_device rd{};
const auto seed = rd();
std::mt19937 rng(seed);

using Device = Eigen::DefaultDevice;
Device device;

const Tensor_3D batched_matrix_multiplication(const Tensor_3D& A, const Tensor_3D& B)
{
    const int batch_size = A.dimension(0);
    const int dim1 = A.dimension(1);
    const int dim2 = B.dimension(2);
    Tensor_3D output(batch_size, dim1, dim2);
    const std::array<Eigen::IndexPair<int>, 1> dims = {Eigen::IndexPair<int>(1, 0)};
    for (int i = 0; i < batch_size; ++i) {
        auto B_chip = B.template chip<0>(i);
        output.template chip<0>(i) = A.template chip<0>(i).contract(B_chip, dims);
    }
    return output;
}

template <typename Activation>
auto gradient(const Tensor_2D &dC_dY, const Tensor_2D &input,
               const Tensor_2D &Z, const Tensor_2D &Y, const Tensor_2D &W,
               const Activation &activation, const bool propagate = true)
{
    const int batch_size = input.dimension(0);
    // calculating dY_dZ

    Tensor_3D dY_dZ = activation.jacobian(Z);

    // reshaping dC_dY1 to 3D to meet BMM 
    const DimArray<3> dC_dY_3D_dim = {batch_size, 1, Y.dimension(1)};
    Tensor_3D dC_dY_3D = dC_dY.reshape(dC_dY_3D_dim);

    // calculating dC_dZ
    Tensor_3D dC_dZ = batched_matrix_multiplication(dC_dY_3D, dY_dZ);

    // calculating dC_dW1, aka, grad1
    const std::array<Eigen::IndexPair<int>, 1> product_dims_0_0 = {Eigen::IndexPair<int>(0, 0)};
    const DimArray<2> dC_dW_dim = {W.dimension(0), W.dimension(1)};
    Tensor_2D dC_dW = input.contract(dC_dZ, product_dims_0_0).reshape(dC_dW_dim);

    Tensor_2D downstream;
    if (propagate) { //false only for the first hidden layer
        // calculating the error propagation dC_dY for the previous layer
        const DimArray<2> error_propagation_dim = {batch_size, input.dimension(1)};
        const std::array<Eigen::IndexPair<int>, 1> product_dims_2_1 = {Eigen::IndexPair<int>(2, 1)};
        downstream = dC_dZ.contract(W, product_dims_2_1).reshape(error_propagation_dim);
    }

    return std::make_tuple(dC_dW, downstream);
}

auto backward(const Tensor_2D &TRUE, const Tensor_2D &X,
               const Tensor_2D &Z0, const Tensor_2D &Z1, const Tensor_2D &Z2,
               const Tensor_2D &Y0, const Tensor_2D &Y1, const Tensor_2D &Y2,
               const Tensor_2D &W0, const Tensor_2D &W1, const Tensor_2D &W2)
{
    
    const int batch_size = TRUE.dimension(0);

    // First step, calculating the output loss gradient dC_dY2

    CategoricalCrossEntropy cost_fn;
    auto dC_dY2 = cost_fn.derivative(TRUE, Y2);

    // second step: calculating weight gradients

    Softmax<Device> softmax(device);
    ReLU<Device, 2> relu(device);

    auto [grad2, dC_dY1] = gradient(dC_dY2, Y1, Z2, Y2, W2, softmax);

    auto [grad1, dC_dY0] = gradient(dC_dY1, Y0, Z1, Y1, W1, relu);

    auto [grad0, _]      = gradient(dC_dY0, X,  Z0, Y0, W0, relu, false);

    return std::make_tuple(grad0, grad1, grad2);
}

auto forward(const Tensor_2D &X, const Tensor_2D &W0, const Tensor_2D &W1, const Tensor_2D &W2)
{
    const std::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(1, 0)};

    // First Hidden Layer
    ReLU<Device, 2> relu(device);
    Tensor_2D Z0 = X.contract(W0, contract_dims);
    Tensor_2D Y0 = relu.evaluate(Z0);

    // Second Hidden Layer
    Tensor_2D Z1 = Y0.contract(W1, contract_dims);
    Tensor_2D Y1 = relu.evaluate(Z1);

    // Output Layer
    Softmax<Device> softmax(device);
    Tensor_2D Z2 = Y1.contract(W2, contract_dims);
    auto Y2 = softmax.evaluate(Z2);

    return std::make_tuple(Z0, Z1, Z2, Y0, Y1, Y2);
}

void update(Tensor_2D &W0,Tensor_2D &W1,Tensor_2D &W2, 
            Tensor_2D &grad0,Tensor_2D &grad1,Tensor_2D &grad2, 
            const TYPE learning_rate)
{
    W0 = W0 - grad0 * grad0.constant(learning_rate);
    W1 = W1 - grad1 * grad1.constant(learning_rate);
    W2 = W2 - grad2 * grad2.constant(learning_rate);
}

TYPE loop(const Tensor_2D &TRUE, const Tensor_2D &X, 
       Tensor_2D &W0,Tensor_2D &W1,Tensor_2D &W2, 
       const TYPE learning_rate)
{
    
    // forward pass
    auto [Z0, Z1, Z2, Y0, Y1, Y2] = forward(X, W0, W1, W2);

    // Output cost
    CategoricalCrossEntropy cost_fn;
    TYPE LOSS = cost_fn.evaluate(TRUE, Y2);

    // backward pass
    auto [grad0, grad1, grad2] = backward(TRUE, X, Z0, Z1, Z2, Y0, Y1, Y2, W0, W1, W2);

    // update pass
    update(W0, W1, W2, grad0, grad1, grad2, learning_rate);

    return LOSS;
}

Tensor_2D predict(const Tensor_2D &TRUE, const Tensor_2D &X,Tensor_2D &W0,Tensor_2D &W1,Tensor_2D &W2)
{
    const std::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(1, 0)};
    // Layer 0
    Tensor_2D Z0 = X.contract(W0, contract_dims);
    ReLU<Device, 2> relu(device);
    Tensor_2D Y0 = relu.evaluate(Z0);

    // Output Layer
    Tensor_2D Z1 = Y0.contract(W1, contract_dims);
    auto Y1 = relu.evaluate(Z1);

    // Output Layer
    Tensor_2D Z2 = Y1.contract(W2, contract_dims);
    Softmax<Device> softmax(device);
    auto Y2 = softmax.evaluate(Z2);

    return Y2;
}

int main(int, char **)
{
    
    auto [training_images, training_labels] = load_mnist("../../data/mnist", rng);
    std::cout << "Data loaded!\n";
    std::cout << "training_images: " << training_images.dimensions() << "\n\n";
    std::cout << "training_labels: " << training_labels.dimensions() << "\n\n";

    const int input_size = training_images.dimension(1);
    const int output_size = training_labels.dimension(1);

    const int hidden_units_0 = 128;
    const int hidden_units_1 = 64;

    Tensor_2D W0(input_size, hidden_units_0);
    glorot_uniform_initializer(rng, W0);
    Tensor_2D W1(hidden_units_0, hidden_units_1);
    glorot_uniform_initializer(rng, W1);
    Tensor_2D W2(hidden_units_1, output_size);
    glorot_uniform_initializer(rng, W2);

    const int MAX_EPOCHS = 20;
    const float learning_rate = 0.1f;

    int epoch = 0;

    while (epoch++ < MAX_EPOCHS)
    {

        auto begin = high_resolution_clock::now();

        float training_loss = loop(training_labels, training_images, W0, W1, W2, learning_rate);

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - begin);

        auto training_pred = predict(training_labels, training_images, W0, W1, W2);
        float training_acc = accuracy(training_labels, training_pred);

        std::cout << "epoch:\t" << epoch
                    << "\ttook:\t" << duration.count() << " mills\t"
                    << "\ttraining_loss:\t" << training_loss
                    << "\ttraining_acc:\t" << training_acc
                    << "\n";

    }

    return 0;
}