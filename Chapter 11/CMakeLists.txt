cmake_minimum_required(VERSION 3.8)

set(PROJECT_NAME chapte_11)
project(${PROJECT_NAME} CXX)

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
find_package(Boost COMPONENTS filesystem system iostreams REQUIRED)

include_directories(include)

add_compile_options(-march=native -mavx2 -mfma)
link_libraries(Eigen3::Eigen)
link_libraries(${CMAKE_THREAD_LIBS_INIT})
link_libraries(TBB::tbb)
link_libraries(${OpenCV_LIBS})
link_libraries(${Boost_LIBRARIES})

add_executable(fitting_polynoms "${PROJECT_SOURCE_DIR}/src/fitting_polynoms.cpp")

file(GLOB overfitting_example_SOURCES_LIB "${PROJECT_SOURCE_DIR}/src/lib/*.cpp" ${PROJECT_SOURCE_DIR}/src/overfitting_example.cpp)
add_executable(overfitting_example ${overfitting_example_SOURCES_LIB})

file(GLOB penalization_SOURCES_LIB "${PROJECT_SOURCE_DIR}/src/lib/*.cpp" ${PROJECT_SOURCE_DIR}/src/penalization.cpp)
add_executable(penalization ${penalization_SOURCES_LIB})

file(GLOB data_augmentation_SOURCES_LIB "${PROJECT_SOURCE_DIR}/src/lib/*.cpp" ${PROJECT_SOURCE_DIR}/src/data_augmentation.cpp)
add_executable(data_augmentation ${data_augmentation_SOURCES_LIB})

file(GLOB dropout_SOURCES_LIB "${PROJECT_SOURCE_DIR}/src/lib/*.cpp" ${PROJECT_SOURCE_DIR}/src/dropout_example.cpp)
add_executable(dropout_example ${dropout_SOURCES_LIB})
