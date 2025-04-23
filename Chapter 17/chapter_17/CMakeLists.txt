cmake_minimum_required(VERSION 3.8)

set(PROJECT_NAME chapter_17)
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
link_libraries(Eigen3::Eigen)
link_libraries(${CMAKE_THREAD_LIBS_INIT})
link_libraries(TBB::tbb)
link_libraries(${OpenCV_LIBS})

# experiment 1

file(GLOB regression_model_SOURCES_LIB "${PROJECT_SOURCE_DIR}/src/lib/*.cpp" ${PROJECT_SOURCE_DIR}/src/regression_model.cpp)
add_executable(regression_model ${regression_model_SOURCES_LIB})

# experiment 2

file(GLOB net_training_oxford_pets_SOURCES_LIB "${PROJECT_SOURCE_DIR}/src/lib/*.cpp" ${PROJECT_SOURCE_DIR}/src/net_training_oxford_pets.cpp)
add_executable(net_training_oxford_pets ${net_training_oxford_pets_SOURCES_LIB})

# experiment 3

file(GLOB fine_tuning_vgg_net_training_oxford_pets_SOURCES_LIB "${PROJECT_SOURCE_DIR}/src/lib/*.cpp" ${PROJECT_SOURCE_DIR}/src/fine_tuning_vgg_net_training_oxford_pets.cpp)
add_executable(fine_tuning_vgg_net_training_oxford_pets ${fine_tuning_vgg_net_training_oxford_pets_SOURCES_LIB})
