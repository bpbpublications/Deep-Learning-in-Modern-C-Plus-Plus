cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(functional_programming
        VERSION 1.0
        DESCRIPTION "Examples of C++ programming using Functional Features, algorithm & numeric header"
        LANGUAGES CXX)

# C++17 is mandatory
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(USING_ALGORITHMS_TARGET_NAME "using_algorithms")
add_executable(${USING_ALGORITHMS_TARGET_NAME} "${PROJECT_SOURCE_DIR}/src/${USING_ALGORITHMS_TARGET_NAME}.cpp")

set(USING_FUNCTIONAL_TARGET_NAME "using_functional")
add_executable(${USING_FUNCTIONAL_TARGET_NAME} "${PROJECT_SOURCE_DIR}/src/${USING_FUNCTIONAL_TARGET_NAME}.cpp")

set(USING_NUMERIC_TARGET_NAME "using_numeric")
add_executable(${USING_NUMERIC_TARGET_NAME} "${PROJECT_SOURCE_DIR}/src/${USING_NUMERIC_TARGET_NAME}.cpp")
