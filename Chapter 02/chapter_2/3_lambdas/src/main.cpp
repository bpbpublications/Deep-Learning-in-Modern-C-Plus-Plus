#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

using VectorType = std::vector<double>;

int main() {
    auto L2 = [](const VectorType &vec) { return std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0); };

    VectorType weights{1., 2., 3., 4., 5., 6.};

    std::cout << "L2(weights) = " << L2(weights) << "\n";

    // Data members of classes (but not structs) additionally have trailing underscores.
    auto momentum_optimizer = [vec_ = VectorType{}](const VectorType &gradient) mutable {
        if (vec_.empty()) {
            vec_.resize(gradient.size());
        }
        std::transform(vec_.begin(), vec_.end(), gradient.begin(), vec_.begin(), [](double v, double dx) {
            double beta = 0.3;
            return beta * v + dx;
        });
        return vec_;
    };

    auto print = [](double d) { std::cout << d << " "; };

    const VectorType current_grads{1., 0., 1., 1., 0., 1.};
    for (int i = 0; i < 3; ++i) {
        VectorType weight_update = momentum_optimizer(current_grads);
        std::for_each(weight_update.begin(), weight_update.end(), print);
        std::cout << "\n";
    }

    return 0;
}
