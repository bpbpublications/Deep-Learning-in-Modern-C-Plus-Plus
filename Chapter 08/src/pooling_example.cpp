#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

#include "book/definitions.hpp"

const Tensor_2D pooling(const Tensor_2D &X, int pool_size, int strides = 1) {

    // reshaping the input to the 4-Rank shape (depth, rows, columns, batch)
    DimArray<4> reshaped_dims{{1, X.dimension(0), X.dimension(1), 1}};
    auto reshaped = X.reshape(reshaped_dims);

    // getting the patches
    Tensor<5> patches = reshaped.extract_image_patches(pool_size, pool_size, strides, strides, Eigen::PADDING_VALID);

    // getting the max of each patch
    DimArray<2> dims({1, 2});
    auto max_patches = patches.maximum(dims);
    //auto max_patches = patches.mean(dims);

    // reshaping back to fit 2-RANK dimensions. Note that extract_image_patches visit each row/col
    // Thus, there is an automatic padding in the left & bottom ends
    int pre_rows = X.dimension(0) / strides;
    int pre_cols = X.dimension(1) / strides;
    DimArray<2> pre_dims{{pre_rows, pre_cols}};
    auto pre = max_patches.reshape(pre_dims);

    int rows = (X.dimension(0) - pool_size) / strides + 1;
    int cols = (X.dimension(1) - pool_size) / strides + 1;

    DimArray<2> offsets = {0, 0};
    DimArray<2> extents = {rows, cols};
    Tensor_2D result = pre.slice(offsets, extents);

    return result;
}

int main(int, char**)
{

    Tensor_2D input(6, 8);
    input.setValues({
                { 0.f,  1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f},
                { 8.f,  9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f},
                {16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f},
                {24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.f},
                {32.f, 33.f, 34.f, 35.f, 36.f, 37.f, 38.f, 39.f},
                {40.f, 41.f, 42.f, 43.f, 44.f, 45.f, 46.f, 47.f}
            });

    auto res_strides_1 = pooling(input, 2);
    std::cout << "res_strides_1: \n\n" << res_strides_1 << "\n\n";

    auto res_strides_2 = pooling(input, 2, 2);
    std::cout << "res_strides_2: \n\n" << res_strides_2 << "\n\n";

    return 0;
}
