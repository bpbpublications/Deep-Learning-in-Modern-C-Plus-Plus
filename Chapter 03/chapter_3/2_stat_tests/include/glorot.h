#ifndef STAT_TESTS_INCLUDE_GLOROT_H_
#define STAT_TESTS_INCLUDE_GLOROT_H_

#include <vector>

float s(float z);

int find_median(const std::vector<int> &arr);

std::vector<float> glorot_initializer(int fan_in, int fan_out);

#endif // STAT_TESTS_INCLUDE_GLOROT_H_
