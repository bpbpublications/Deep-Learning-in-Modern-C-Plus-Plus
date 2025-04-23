#include "glorot.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

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
