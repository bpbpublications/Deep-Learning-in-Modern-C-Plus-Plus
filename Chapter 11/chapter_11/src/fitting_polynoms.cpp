#include <numeric>
#include <iostream>
#include <random>
#include <Eigen/Dense>

double MSE(const Eigen::VectorXd & TRUE, const Eigen::VectorXd & PRED) {
    auto diff = TRUE - PRED;
    auto power = diff.array().pow(2.);
    double sum = power.sum();
    double result = sum / PRED.size();
    return result;
}

Eigen::VectorXd fit_polynom(const Eigen::VectorXd& X, const Eigen::VectorXd& Y, const int degree) {

    Eigen::MatrixXd terms = Eigen::MatrixXd::Ones(X.size(), degree + 1);
    for (int i = 1; i <= degree; ++i)
    {
        terms.col(i) = terms.col(i - 1).cwiseProduct(X);
    }
    auto solver = terms.householderQr();
    auto coeffs = solver.solve(Y);
    return coeffs; 
}

Eigen::VectorXd eval(const Eigen::VectorXd& coeffs, const Eigen::VectorXd& X) {

    const int degree = coeffs.size() - 1;

    Eigen::MatrixXd terms = Eigen::MatrixXd::Ones(X.size(), degree + 1);
    for (int i = 1; i <= degree; ++i)
    {
        terms.col(i) = terms.col(i - 1).cwiseProduct(X);
    }

    Eigen::VectorXd result = terms * coeffs;

    return result;
}

int main(int, char**)
{
    double range = 1.;
    std::random_device rd {};
    //auto seed = rd();
    long seed = 900027959;
    std::mt19937 rng(seed);
    srand((unsigned int) seed);

    auto generate_synthetic_data = [&](int size = 10, double noise_dev = 0.3) {

        std::uniform_real_distribution<double> uniform_distro(0, range);
        Eigen::VectorXd X = Eigen::VectorXd::Zero(size).unaryExpr([&](double){return uniform_distro(rng);});
        std::sort(std::begin(X), std::end(X));

        Eigen::VectorXd G = X.unaryExpr([&](double x){return sin(2.*M_PI*x);});
        
        std::normal_distribution<double> normal_distro(0, noise_dev);
        Eigen::VectorXd NOISE = Eigen::VectorXd::Zero(size).unaryExpr([&](double){return normal_distro(rng);});

        Eigen::VectorXd TRUE = G + NOISE;

        return std::make_pair(X, TRUE);
    };

    const auto [X, TRUE] = generate_synthetic_data();

    std::vector<int> poly_degrees{1, 2, 3, 4, 5, 6, 7, 8, 9};

    Eigen::MatrixXd results = Eigen::MatrixXd::Zero(11, poly_degrees.size());

    Eigen::VectorXd input = Eigen::VectorXd::LinSpaced(results.rows(), 0., range);

    for (unsigned i = 0; i < poly_degrees.size(); ++i)
    {
        int degree = poly_degrees[i];
        auto params = fit_polynom(X, TRUE, degree);

        auto Y = eval(params, input);

        results.col(i) = Y;

        auto PRED = eval(params, X);

        // std::cout << "polynom degree " << degree << " mse is " << MSE(TRUE, PRED) << "\n";

    }

    Eigen::IOFormat formatter(4, 0, "\t", "\n", "", "");
    for (int degree : poly_degrees) std::cout << "poly degree " << degree << "\t";
    std::cout << "\n" << results.format(formatter) << "\n\n";

    return 0;

}