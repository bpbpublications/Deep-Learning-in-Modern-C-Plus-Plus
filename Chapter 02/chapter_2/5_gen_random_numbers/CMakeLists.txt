cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(stats_cpp
        VERSION 1.0
        DESCRIPTION "Gen random numbers"
        LANGUAGES CXX)

# C++17 is mandatory
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/using_stat_distros.cpp)
