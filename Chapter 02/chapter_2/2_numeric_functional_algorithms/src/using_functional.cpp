#include <algorithm>  // std::for_each
#include <functional> // std::less, std::less_equal, std::greater, std::greater_equal
#include <iostream>   // std::cout
#include <vector>     // std::vector

int main() {
    std::vector<std::function<bool(double, double)>> comparators{std::less<>(), std::less_equal<>(), std::greater<>(),
                                                                 std::greater_equal<>()};

    double x = 10.;
    double y = 10.;

    auto compare = [&x, &y](const std::function<bool(double, double)> &comparator) {
        bool b = comparator(x, y);
        std::cout << (b ? "TRUE" : "FALSE") << "\n";
    };

    std::for_each(comparators.begin(), comparators.end(), compare);

    return 0;
}
