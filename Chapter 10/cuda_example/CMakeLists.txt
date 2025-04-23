cmake_minimum_required(VERSION 3.8)

project(cuda_helloworld)
enable_language("CUDA")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

include_directories(include)
link_libraries(Eigen3::Eigen)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/src/main.cc ${PROJECT_SOURCE_DIR}/src/functions.cu)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})