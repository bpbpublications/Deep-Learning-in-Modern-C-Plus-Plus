#include <gtest/gtest.h>

#include "my_functions.h"

TEST(OddCase, BaseCases) {
    std::vector<int> vec{9, 4, 1, 11, 20, -1, 2};
    int actual = find_middle(vec);

    EXPECT_EQ(4, actual);
}

TEST(EvenCase, BaseCases) {
    std::vector<int> vec{9, 4, 11, 20, -1, 2};
    int actual = find_middle(vec);

    EXPECT_EQ(9, actual);
}

TEST(CheckEmptyVector, BaseCases) {

    EXPECT_THROW(
        {
            std::vector<int> vec; // an empty vector
            find_middle(vec);
        },
        std::invalid_argument);
}
