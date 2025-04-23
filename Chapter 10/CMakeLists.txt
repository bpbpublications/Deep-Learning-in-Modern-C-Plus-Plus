cmake_minimum_required(VERSION 3.8)

set(PROJECT_NAME chapter_10)
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
find_package(TBB REQUIRED)

include_directories(include)

link_libraries(Eigen3::Eigen)
link_libraries(${CMAKE_THREAD_LIBS_INIT})
link_libraries(TBB::tbb)
link_libraries(${OpenCV_LIBS})

file(GLOB SLOW_EXAMPLE_SOURCES_LIB "${PROJECT_SOURCE_DIR}/src/lib/*.cpp" ${PROJECT_SOURCE_DIR}/src/naive_backpropagation.cpp)
add_executable(naive_backpropagation ${SLOW_EXAMPLE_SOURCES_LIB})

file(GLOB FAST_EXAMPLE_SOURCES_LIB "${PROJECT_SOURCE_DIR}/src/lib/*.cpp" ${PROJECT_SOURCE_DIR}/src/improving_backpropagation.cpp)
add_executable(improving_backpropagation ${FAST_EXAMPLE_SOURCES_LIB})

if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  target_compile_options(naive_backpropagation PRIVATE -mavx2 -mfma)
  target_compile_options(improving_backpropagation PRIVATE -mavx2 -mfma)
endif() 
