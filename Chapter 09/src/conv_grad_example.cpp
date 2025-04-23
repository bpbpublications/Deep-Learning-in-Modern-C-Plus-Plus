#include <iostream>

#include <unsupported/Eigen/CXX11/Tensor>

auto convolution2D = [](const Eigen::Tensor<float, 2> &input, const Eigen::Tensor<float, 2> &kernel, int pad = 0)
{

    if (pad > 0) {

        Eigen::array<std::pair<int, int>, 2> padding;
        padding[0] = std::make_pair(pad, pad);
        padding[1] = std::make_pair(pad, pad);
        auto padded = input.pad(padding);

        Eigen::array<int, 2> dims({0, 1});
        Eigen::Tensor<float, 2> result  = padded.convolve(kernel, dims);
        return result;

    } else {

        Eigen::array<int, 2> dims({0, 1});
        Eigen::Tensor<float, 2> result  = input.convolve(kernel, dims);
        return result;

    }

};

auto rotate180 = [](const Eigen::Tensor<float, 2> &tensor){
    Eigen::array<bool, 2> reverse({true, true});
    Eigen::Tensor<float, 2> result = tensor.reverse(reverse);
    return result;
};

int main(int, char**) {

    Eigen::Tensor<float, 2> X(6, 5);
    X.setValues({
        {1, 0, -1, 1, 1}, 
        {5, 2, 1, -1, -3}, 
        {1, 3, 1, -1, 3}, 
        {2, 2, 2, 2, 3},
        {3, -3, 1, 3, -1},
        {2, -1, 2, -2, 1}
    });

    Eigen::Tensor<float, 2> dC_dO(4, 3);
    dC_dO.setValues({
        {2., 2., 2.}, 
        {2., 2., 2.}, 
        {2., 2., 2.}, 
        {2., 2., 2.}
    });

    auto dC_dK = convolution2D(X, dC_dO);

    std::cout << "dC_dK:\n" << dC_dK << "\n";

    Eigen::Tensor<float, 2> kernel(3, 3);
    kernel.setValues({
        {1, 0, -2 }, 
        {2, 1, -2}, 
        {-3, -2, 3}
    });

    auto kernel_180 = rotate180(kernel);

    std::cout << "\nkernel_180:\n" << kernel_180 << "\n";

    auto dC_dX = convolution2D(dC_dO, kernel_180, 2);

    std::cout << "\ndC_dX:\n" << dC_dX << "\n";

    return 0;
}