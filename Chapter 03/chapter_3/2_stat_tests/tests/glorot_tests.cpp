#include <gtest/gtest.h>

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/students_t.hpp>

#include <cstddef>
#include <numeric>

#include "glorot.h"

/**
 * The NULL hypothesis states sample_std eq population_std
 *
 * This function returns true if the NULL hypothesis was rejected
 */
bool chi_squared_test(float population_std, float sample_std, int sample_size, float alpha) {
    float variance_rate = sample_std / population_std;
    variance_rate = variance_rate * variance_rate;
    float t_stat = static_cast<float>(sample_size - 1) * variance_rate;

    boost::math::chi_squared distro(sample_size - 1);

    const auto upper_limit = static_cast<float>(quantile(complement(distro, alpha / 2)));
    const auto lower_limit = static_cast<float>(quantile(distro, alpha / 2));

    return t_stat > upper_limit && t_stat < lower_limit;
}

/**
 * The NULL hypothesis states sample_mean eq population_mean
 *
 * This function returns true if the NULL hypothesis was rejected
 */
bool t_test(float population_mean, float sample_mean, float sample_std_dev, int sample_size, float alpha) {
    float diff = sample_mean - population_mean;

    auto t_stat = static_cast<float>(diff * sqrt(double(sample_size)) / sample_std_dev);

    unsigned degree_of_freedom = sample_size - 1;
    boost::math::students_t distro(degree_of_freedom);
    const auto q = static_cast<float>(cdf(complement(distro, fabs(t_stat))));

    const auto alpha_2 = static_cast<float>(alpha / 2.);

    return q < alpha_2;
}

TEST(CheckGlorot, StatCases) {
    int fan_in = 6;
    int fan_out = 5;

    auto weigths = glorot_initializer(fan_in, fan_out);

    const std::size_t N = weigths.size();

    EXPECT_EQ(N, fan_in * fan_out);

    const auto sum = static_cast<float>(std::accumulate(weigths.begin(), weigths.end(), 0.0));
    float weight_mean = sum / static_cast<float>(N);

    float acc = 0.0;

    auto differ = [&acc, &weight_mean](const float val) {
        const float diff = val - weight_mean;
        acc += diff * diff;
    };

    std::for_each(weigths.begin(), weigths.end(), differ);

    float actual_stdev = std::sqrt(acc / static_cast<float>(N - 1));

    const auto expected_stddev = static_cast<float>(std::sqrt(2. / (fan_in + fan_out)));

    bool std_dev_rejected = chi_squared_test(expected_stddev, actual_stdev, static_cast<int>(N), 0.05);

    if (std_dev_rejected) {
        FAIL() << "The weights standard deviation do not look like expected by the Glorot initializer";
    }

    bool mean_rejected = t_test(0., weight_mean, actual_stdev, static_cast<int>(N), 0.05);

    if (mean_rejected) {
        FAIL() << "The weights mean do not look like expected by the Glorot initializer";
    }
}
