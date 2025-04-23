#include <iostream>

#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/AutoDiff>

typedef typename Eigen::AutoDiffScalar<Eigen::VectorXf> AutoDiff_T;

template <typename T>
T custom_function(const Eigen::Tensor<T, 2> &X, const Eigen::Tensor<T, 2> &K)
{
    Eigen::array<int, 2> dims({0, 1});
    auto conv = X.convolve(K, dims);
    Eigen::Tensor<T, 2> output = conv * conv;
    auto loss = output.sum();
    T result = ((Eigen::Tensor<T, 0>)loss)(0);
    return result;
};

auto convert = [](const Eigen::Tensor<float, 2> &tensor, int offset, int size)
{
    const int rows = tensor.dimension(0);
    const int cols = tensor.dimension(1);

    Eigen::Tensor<AutoDiff_T, 2> result(rows, cols);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            int index = i * cols + j;
            result(i, j).derivatives() = Eigen::VectorXf::Unit(size, offset + index);
            result(i, j).value() = tensor(i, j);
        }
    }

    return result;
};

auto gradients(const AutoDiff_T &Y, const Eigen::Tensor<AutoDiff_T, 2> &X, const Eigen::Tensor<AutoDiff_T, 2> &K)
{

    auto derivatives = Y.derivatives();

    int index = 0;
    Eigen::Tensor<float, 2> dY_dX(X.dimension(0), X.dimension(1));
    for (int i = 0; i < X.dimension(0); ++i)
    {
        for (int j = 0; j < X.dimension(1); ++j)
        {
            float val = derivatives[index];
            dY_dX(i, j) = val;
            index++;
        }
    }

    Eigen::Tensor<float, 2> dY_dK(K.dimension(0), K.dimension(1));
    for (int i = 0; i < K.dimension(0); ++i)
    {
        for (int j = 0; j < K.dimension(1); ++j)
        {
            float val = derivatives[index];
            dY_dK(i, j) = val;
            index++;
        }
    }

    return std::make_pair(dY_dX, dY_dK);
}

int main(int, char **)
{

    Eigen::Tensor<float, 2> x_in(6, 5);
    x_in.setValues({{1, 0, -1, 1, 1},
                    {5, 2, 1, -1, -3},
                    {1, 3, 1, -1, 3},
                    {2, 2, 2, 2, 3},
                    {3, -3, 1, 3, -1},
                    {2, -1, 2, -2, 1}});

    Eigen::Tensor<float, 2> k_in(3, 3);
    k_in.setValues({{1, 0, -2},
                    {2, 1, -2},
                    {-3, -2, 3}});

    Eigen::Tensor<AutoDiff_T, 2> X = convert(x_in, 0, x_in.size() + k_in.size());
    Eigen::Tensor<AutoDiff_T, 2> K = convert(k_in, x_in.size(), x_in.size() + k_in.size());

    auto LOSS = custom_function(X, K);

    auto [dY_dX, dY_dK] = gradients(LOSS, X, K);

    std::cout << "X:\n"
              << X << "\n\n";
    std::cout << "K:\n"
              << K << "\n\n";
    std::cout << "LOSS:\n"
              << LOSS << "\n\n";
    std::cout << "dY_dX:\n"
              << dY_dX << "\n\n";
    std::cout << "dY_dK:\n"
              << dY_dK << "\n\n";

    return 0;
}