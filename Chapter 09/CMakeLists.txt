cmake_minimum_required(VERSION 3.8)

set(PROJECT_NAME chapter_9)
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
find_package(OpenCV REQUIRED)

include_directories(include)

link_libraries(Eigen3::Eigen)
link_libraries(${OpenCV_LIBS})

add_executable(gradient_descent_example "${PROJECT_SOURCE_DIR}/src/gradient_descent_example.cpp")

add_executable(conv_grad_example "${PROJECT_SOURCE_DIR}/src/conv_grad_example.cpp")

add_executable(conv_autograd "${PROJECT_SOURCE_DIR}/src/conv_autograd.cpp")
