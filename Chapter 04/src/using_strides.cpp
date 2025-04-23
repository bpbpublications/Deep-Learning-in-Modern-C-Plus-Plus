#include <iostream>

#include <numeric>

#include <unsupported/Eigen/CXX11/Tensor>

int main(int, char **)
{

    Eigen::Tensor<int, 2> A(4, 5);
    A.setValues({{1, 2, -1, 0, 1},
        {4, 1, 2, -1, 2}, 
        {3, 1, -1, 1, -2}, 
        {-2, 1, 5, 4, 0}
    });
    Eigen::array<Eigen::DenseIndex, 2> strides({3, 2});
    Eigen::Tensor<int, 2> B = A.stride(strides);
    std::cout << "A is\n\n" << A.dimensions() << "\n\n" << A << "\n\n";
    std::cout << "B is\n\n" << B.dimensions() << "\n\n" << B << "\n\n";

    return 0;
}