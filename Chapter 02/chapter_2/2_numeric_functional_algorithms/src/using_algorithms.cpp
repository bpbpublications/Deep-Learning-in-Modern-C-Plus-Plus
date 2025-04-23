#include <algorithm>
#include <iostream>
#include <vector>

double square(double x) { return x * x; }

int main() {
    std::vector<double> X{1., 2., 3., 4., 5., 6.};
    std::vector<double> Y(X.size());

    std::transform(X.begin(), X.end(), Y.begin(), square);

    std::for_each(Y.begin(), Y.end(), [](double y) { std::cout << y << " "; });

    std::cout << "\n";
    return 0;
}
