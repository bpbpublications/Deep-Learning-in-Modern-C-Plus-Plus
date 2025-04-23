cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(using_lambdas
        VERSION 1.0
        DESCRIPTION "Playing with Lambdas"
        LANGUAGES CXX)

# C++17 is mandatory
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

add_executable(${PROJECT_NAME} "${PROJECT_SOURCE_DIR}/src/main.cpp")
