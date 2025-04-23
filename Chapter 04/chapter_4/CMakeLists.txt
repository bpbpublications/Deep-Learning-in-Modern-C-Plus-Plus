cmake_minimum_required(VERSION 3.8)

set(PROJECT_NAME chapter_4)
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

add_executable(convolution_2d_example "${PROJECT_SOURCE_DIR}/src/convolution_2d_example.cpp")
target_link_libraries(convolution_2d_example Eigen3::Eigen)

add_executable(same_padding_example "${PROJECT_SOURCE_DIR}/src/same_padding_example.cpp")
target_link_libraries(same_padding_example Eigen3::Eigen)

add_executable(using_opencv "${PROJECT_SOURCE_DIR}/src/using_opencv.cpp")
target_link_libraries(using_opencv Eigen3::Eigen)
target_link_libraries(using_opencv ${OpenCV_LIBS})

add_executable(applying_sobel "${PROJECT_SOURCE_DIR}/src/applying_sobel.cpp")
target_link_libraries(applying_sobel Eigen3::Eigen)
target_link_libraries(applying_sobel ${OpenCV_LIBS})

add_executable(using_tensors "${PROJECT_SOURCE_DIR}/src/using_tensors.cpp")
target_link_libraries(using_tensors Eigen3::Eigen)

add_executable(tensor_convolutions "${PROJECT_SOURCE_DIR}/src/tensor_convolutions.cpp")
target_link_libraries(tensor_convolutions Eigen3::Eigen)

add_executable(using_strides "${PROJECT_SOURCE_DIR}/src/using_strides.cpp")
target_link_libraries(using_strides Eigen3::Eigen)

add_executable(using_padding "${PROJECT_SOURCE_DIR}/src/using_padding.cpp")
target_link_libraries(using_padding Eigen3::Eigen)
