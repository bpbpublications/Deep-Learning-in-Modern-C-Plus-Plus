cmake_minimum_required(VERSION 3.8)

set(PROJECT_NAME chapter_5)
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

include_directories(include)

add_executable(activation_example "${PROJECT_SOURCE_DIR}/src/sigmoid.cpp")
target_link_libraries(activation_example Eigen3::Eigen)

add_executable(FC_layer_func "${PROJECT_SOURCE_DIR}/src/fc_layer_func.cpp")
target_link_libraries(FC_layer_func Eigen3::Eigen)

add_executable(mlp_example "${PROJECT_SOURCE_DIR}/src/mlp_example.cpp")
target_link_libraries(mlp_example Eigen3::Eigen)

add_executable(using_flatten "${PROJECT_SOURCE_DIR}/src/using_flatten.cpp")
target_link_libraries(using_flatten Eigen3::Eigen)
