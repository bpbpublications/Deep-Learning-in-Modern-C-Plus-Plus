#include <Eigen/Dense>

#include <iostream>

int main() {
    Eigen::MatrixXd A(2, 2);
    A(0, 0) = 2.;
    A(1, 0) = -2.;
    A(0, 1) = 3.;
    A(1, 1) = 1.;

    Eigen::MatrixXd B(2, 3);
    B(0, 0) = 1.;
    B(1, 0) = 1.;
    B(0, 1) = 2.;
    B(1, 1) = 2.;
    B(0, 2) = -1.;
    B(1, 2) = 1.;

    auto C = A * B;

    std::cout << "A:\n" << A << "\n";
    std::cout << "B:\n" << B << "\n";
    std::cout << "C:\n" << C << "\n";

    auto D = B.cwiseProduct(C);
    std::cout << "coeficient-wise multiplication of B & C is:\n" << D << "\n";

    auto E = B + C;
    std::cout << "The sum of B & C is:\n" << E << "\n";

    std::cout << "The transpose of B is:\n" << B.transpose() << "\n";

    std::cout << "The A inverse is:\n" << A.inverse() << "\n";

    std::cout << "The determinant of A is:\n" << A.determinant() << "\n";

    std::cout << "Example of unary operation:\n";
    auto func_X_X = [](double x) { return x * x; };
    std::cout << A.unaryExpr(func_X_X) << "\n";

    std::cout << "Example of binary operation:\n";
    auto func_X_Y = [](double x, double y) { return x * y; };
    std::cout << B.binaryExpr(C, func_X_Y) << "\n";

    return 0;
}
