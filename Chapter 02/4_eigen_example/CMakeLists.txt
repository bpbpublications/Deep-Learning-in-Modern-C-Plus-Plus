cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(eigen_premier
        VERSION 1.0
        DESCRIPTION "Eigen premier"
        LANGUAGES CXX)

# C++17 is mandatory
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

add_executable(${PROJECT_NAME} "${PROJECT_SOURCE_DIR}/src/main.cpp")

# include Eigen3
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
