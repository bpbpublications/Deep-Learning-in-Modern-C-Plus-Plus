cmake_minimum_required(VERSION 3.8)

set(PROJECT_NAME test_googletest_example)
project(${PROJECT_NAME} CXX)

if (NOT CMAKE_BUILD_TYPE) 
  set(CMAKE_BUILD_TYPE Release) 
endif() 

# adding sanitizer may help finding memory bugs
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
  message(STATUS "adding sanitizer")
endif()

# c++17 is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# dependencies
find_package(Eigen3 REQUIRED)

# Locate GTest
enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

include_directories(include)

file(GLOB_RECURSE TEST_SRC_FILES "${PROJECT_SOURCE_DIR}/tests/*.cpp" "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_executable(${PROJECT_NAME} ${TEST_SRC_FILES})

# Link with both gtest and gtest_main Google Test libraries
target_link_libraries(${PROJECT_NAME} ${GTEST_BOTH_LIBRARIES})
gtest_discover_tests(${PROJECT_NAME})

# include PTHREAD
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} Threads::Threads)
