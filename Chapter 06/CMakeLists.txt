cmake_minimum_required(VERSION 3.8)

set(PROJECT_NAME chapter_6)
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

add_executable(h0_h1_example "${PROJECT_SOURCE_DIR}/src/h0_h1_example.cpp")
target_link_libraries(h0_h1_example Eigen3::Eigen)

add_executable(fitting_example "${PROJECT_SOURCE_DIR}/src/fitting_example.cpp")
target_link_libraries(fitting_example Eigen3::Eigen)

add_executable(using_gradient "${PROJECT_SOURCE_DIR}/src/using_gradient.cpp")
target_link_libraries(using_gradient Eigen3::Eigen)

add_executable(bce_example "${PROJECT_SOURCE_DIR}/src/bce_example.cpp")
target_link_libraries(bce_example Eigen3::Eigen)