#include "my_functions.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

float sigmoid(float z) {
    float result;
    if (z >= 45.f) {
        result = 1.f;
    } else if (z <= -45.f) {
        result = 0.f;
    } else {
        result = 1.f / (1.f + std::exp(-z));
    }
    return result;
}

int find_middle(std::vector<int> arr) {
    const std::size_t N = arr.size();
    if (N == 0) {
        throw std::invalid_argument{"cannot find the middle of an empty vector"};
    }
    std::sort(arr.begin(), arr.end());
    int result = arr[N / 2];
    return result;
}

std::vector<float> glorot_initializer(int fan_in, int fan_out) {

    const std::size_t size = fan_in * fan_out;
    const auto stddev = static_cast<float>(std::sqrt(2. / (fan_in + fan_out)));
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<float> distribution(.0, stddev);
    std::vector<float> result(size);
    std::generate(result.begin(), result.end(), [&generator, &distribution]() { return distribution(generator); });
    return result;
}
