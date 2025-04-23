#ifndef GOOGLETEST_SETUP_INCLUDE_MY_FUNCTIONS_H_
#define GOOGLETEST_SETUP_INCLUDE_MY_FUNCTIONS_H_

#include <vector>

float sigmoid(float z);

int find_middle(std::vector<int> arr);

std::vector<float> glorot_initializer(int fan_in, int fan_out);

#endif // GOOGLETEST_SETUP_INCLUDE_MY_FUNCTIONS_H_
