#include <iostream>

#include <Eigen/Core>

#include <cmath>

using Matrix = Eigen::MatrixXd;

int main(int, char **) {

    auto Convolution2D = [](const Matrix &input, const Matrix &kernel, int padding){

        int kernel_rows = kernel.rows();
        int kernel_cols = kernel.cols();

        int rows = input.rows() - kernel_rows + 2*padding + 1;
        int cols = input.cols() - kernel_cols + 2*padding + 1;

        Matrix padded = Matrix::Zero(input.rows() + 2*padding, input.cols() + 2*padding);
        padded.block(padding, padding, input.rows(), input.cols()) = input;

        Matrix result = Matrix::Zero(rows, cols);
        for(int i = 0; i < rows; ++i) {
            for(int j = 0; j < cols; ++j) {
                double sum = padded.block(i, j, kernel_rows, kernel_cols).cwiseProduct(kernel).sum();
                result(i, j) = sum;
            }
        }
        return result;
    }; 

    auto Convolution2D_v2 = [](const Matrix &input, const Matrix &kernel, int padding){

        const int input_rows = input.rows();
        const int input_cols = input.cols();

        const int kernel_rows = kernel.rows();
        const int kernel_cols = kernel.cols();

        if (input_rows < kernel_rows) throw std::invalid_argument("The input has less rows than the kernel");
        if (input_cols < kernel_cols) throw std::invalid_argument("The input has less columns than the kernel");

        const int rows = input_rows - kernel_rows + 2*padding + 1;
        const int cols = input_cols - kernel_cols + 2*padding + 1;

        Matrix result = Matrix::Zero(rows, cols);

        auto fit_dims = [&padding](int pos, int k, int length) {
            int input = pos - padding;
            int kernel = 0;
            int size = k;

            if (input < 0) {
                kernel = -input;
                size += input;
                input = 0;
            }

            if (input + size > length) {
                size = length - input;
            }

            return std::make_tuple(input, kernel, size);
        };

        for(int i = 0; i < rows; ++i) {

            const auto [input_i, kernel_i, size_i] = fit_dims(i, kernel_rows, input_rows);

            for(int j = 0; size_i > 0 && j < cols; ++j) {

                const auto [input_j, kernel_j, size_j] = fit_dims(j, kernel_cols, input_cols);

                if (size_j > 0) {
                    auto input_tile = input.block(input_i, input_j, size_i, size_j);
                    auto input_kernel = kernel.block(kernel_i, kernel_j, size_i, size_j);
                    result(i, j) = input_tile.cwiseProduct(input_kernel).sum();
                }
            }
        }
        return result;
    }; 

    Matrix input(6, 6);
    input << 
        3, 1, 0, 2, 5, 6,
        4, 2, 1, 1, 4, 7,
        5, 4, 0, 0, 1, 2,
        1, 2, 2, 1, 3, 4,
        6, 3, 1, 0, 5, 2,
        3, 1, 0, 1, 3, 3;

    Matrix kernel3(3, 3);

    kernel3 << 
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1;

    auto out = Convolution2D_v2(input, kernel3, 1);

    std::cout << "Kernel:\n" << kernel3 << "\n\n";
    std::cout << "Input:\n" << input << "\n\n";
    std::cout << "Convolution:\n" << out << "\n";

    return 0;
}