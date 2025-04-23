cmake_minimum_required(VERSION 3.8)

set(PROJECT_NAME chapter_12)
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

add_compile_options(-march=native -mavx2 -mfma)

file(GLOB stratification_SOURCES_LIB "${PROJECT_SOURCE_DIR}/src/lib/*.cpp" ${PROJECT_SOURCE_DIR}/src/stratification.cpp)
add_executable(stratification ${stratification_SOURCES_LIB})
target_link_libraries(stratification Eigen3::Eigen)
target_link_libraries(stratification ${OpenCV_LIBS})

file(GLOB metrics_SOURCES_LIB "${PROJECT_SOURCE_DIR}/src/lib/*.cpp" ${PROJECT_SOURCE_DIR}/src/using_metrics.cpp)
add_executable(using_metrics ${metrics_SOURCES_LIB})
target_link_libraries(using_metrics Eigen3::Eigen)
target_link_libraries(using_metrics ${OpenCV_LIBS})

file(GLOB minibatch_SOURCES_LIB "${PROJECT_SOURCE_DIR}/src/lib/*.cpp" ${PROJECT_SOURCE_DIR}/src/minibatch.cpp)
add_executable(minibatch ${minibatch_SOURCES_LIB})
target_compile_options(minibatch PRIVATE -march=native -mavx2 -mfma)
target_link_libraries(minibatch Eigen3::Eigen)
target_link_libraries(minibatch ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(minibatch TBB::tbb)
target_link_libraries(minibatch ${OpenCV_LIBS})

file(GLOB sgd_SOURCES_LIB "${PROJECT_SOURCE_DIR}/src/lib/*.cpp" ${PROJECT_SOURCE_DIR}/src/sgd_example.cpp)
add_executable(sgd_example ${sgd_SOURCES_LIB})
target_compile_options(sgd_example PRIVATE -march=native -mavx2 -mfma)
target_link_libraries(sgd_example Eigen3::Eigen)
target_link_libraries(sgd_example ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(sgd_example TBB::tbb)
target_link_libraries(sgd_example ${OpenCV_LIBS})
