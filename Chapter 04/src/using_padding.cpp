#include <iostream>

#include <numeric>

#include <unsupported/Eigen/CXX11/Tensor>

int main(int, char **)
{

    Eigen::Tensor<int, 2> A(2, 3);
    A.setValues({{0, 100, 200}, {300, 400, 500}});

    Eigen::array<std::pair<int, int>, 2> padding;
    padding[0] = std::make_pair(0, 1);
    padding[1] = std::make_pair(2, 3);
    Eigen::Tensor<int, 2> B = A.pad(padding);

    std::cout << "A is\n\n" << A.dimensions() << "\n\n" << A << "\n\n";
    std::cout << "B is\n\n" << B.dimensions() << "\n\n" << B << "\n\n";

    return 0;
}