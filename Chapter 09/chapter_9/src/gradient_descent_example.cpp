#include <iostream>
#include <chrono>
using namespace std::chrono;

#include <unsupported/Eigen/CXX11/Tensor>

#include "book/definitions.hpp"

#include "image_io.hpp"

auto BatchedConvolution = [](const Tensor_3D &A, const Tensor_3D &B)
{

    const int pad_set = 1;
    Eigen::array<std::pair<int, int>, 3> padding;
    padding[0] = std::make_pair(0, 0);
    padding[1] = std::make_pair(pad_set, pad_set);
    padding[2] = std::make_pair(pad_set, pad_set);

    const int batch_size = A.dimension(0);
    const int dim1 = A.dimension(1);
    const int dim2 = A.dimension(2);
    Tensor_3D output(batch_size, dim1, dim2);

    auto padded = A.pad(padding);
    DimArray<3> conv_dims({0, 1, 2});
    output = padded.convolve(B, conv_dims);
    
    return output;
};

auto backward = [](const Tensor_3D &X, Tensor_3D &T, Tensor_3D &Y)
{
    auto DIFF = Y - T;
    Tensor_3D batch = BatchedConvolution(X, DIFF);

    DimArray<2> two_dims{{batch.dimension(1), batch.dimension(2)}};
    Tensor_2D result = batch.reshape(two_dims);
    result = result * result.constant(2.f / X.size());

    return result;
};

auto convolution2D = [](const Tensor_3D &input, const Tensor_2D &kernel)
{

    Eigen::array<std::pair<int, int>, 3> padding;
    padding[0] = std::make_pair(0, 0);
    padding[1] = std::make_pair(1, 1);
    padding[2] = std::make_pair(1, 1);

    auto padded = input.pad(padding);
    DimArray<2> dims({1, 2});
    Tensor_3D result  = padded.convolve(kernel, dims);
    return result;
};

auto forward = [](const Tensor_3D &X, const Tensor_2D &kernel)
{
    return convolution2D(X, kernel);
};

auto MSE = [](const Tensor_3D &T, const Tensor_3D &Y)
{

    auto diff = T - Y;
    auto quadratic = diff * diff;
    Tensor_0D sum = quadratic.sum();

    float result = sum(0) / Y.size();

    return result;
};

int main(int argc, char**) {

    Tensor_2D Generator_Kernel(3, 3);
    Generator_Kernel.setValues({{1., 0., -1.}, {2., 0., -2.},{1., 0., -1.}});

    auto [X, T] = load_dataset("../images/", Generator_Kernel, 160);

    if (argc > 1) {

        const int batch_size = X.dimension(0);
        const int image_size = X.dimension(1);
        const DimArray<3> image_dim = {image_size, image_size, 1};

        for (int i = 0; i < batch_size; ++i) {
            cv::Mat X_output, T_output;

            Tensor_3D x = X.chip<0>(i).reshape(image_dim);
            cv::eigen2cv(x, X_output);
            cv::imshow("x", X_output);

            Tensor_3D t = T.chip<0>(i).reshape(image_dim);
            Tensor_0D _max = t.maximum();
            Tensor_3D normalization = t / t.constant(_max(0));
            cv::eigen2cv(normalization, T_output);
            cv::imshow("t", T_output);

            cv::waitKey();
        }

        cv::destroyAllWindows();
    }

    Tensor_2D kernel(3, 3);
    kernel = kernel.random();

    const int MAX_EPOCHS = 5'000;
    const double learning_rate = 0.1;

    int epoch = 0;
    while (epoch < MAX_EPOCHS)
    {

        auto begin = high_resolution_clock::now();
        auto output = forward(X, kernel);

        auto grad = backward(X, T, output);

        double loss = MSE(T, output);

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - begin);

        auto update = grad * grad.constant(learning_rate);
        kernel -= update;

        if (epoch % 100 == 0) {
            std::cout << "epoch:\t" << epoch << "\ttook:\t" << duration.count() << " mills\t" << "\tloss:\t" << loss << "\n";
        }

        epoch++;
    }

    std::cout << "\nGenerative kernel is:\n\n" << std::fixed << std::setprecision(2) << Generator_Kernel << "\n\n";
    std::cout << "\nTrained kernel is:\n\n" << std::fixed << std::setprecision(2) << kernel << "\n\n";

    return 0;
}