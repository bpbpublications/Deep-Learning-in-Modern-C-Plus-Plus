#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>

std::random_device rd{};

int main() {
    const unsigned int seed = rd();
    std::mt19937 random_generator {seed};
 
    std::uniform_real_distribution<> distro {-10, 13};
    std::uniform_real_distribution<> distro2 {-5, 8};
    std::uniform_real_distribution<> distro3 {3, 9};
    std::uniform_real_distribution<> distro4 {-8, -1};
    std::uniform_real_distribution<> distro5 {-10, 10};
  
    std::map<int, int> histogram;

    for (int n = 0; n < 10'000; ++n) {
        double val = distro(random_generator);
        val += distro2(random_generator);
        val += distro3(random_generator);
        val += distro4(random_generator);
        val += distro5(random_generator);
        val /= 5.;
        ++histogram[static_cast<int>(std::round(val))];
    }

    for (auto [x, y] : histogram) {
        std::cout << std::setw(2) << x << ' ' << std::string(y / 20, '*') << '\n';
    }

    std::cout << distro(random_generator) << "\n";

    return 0;
}
