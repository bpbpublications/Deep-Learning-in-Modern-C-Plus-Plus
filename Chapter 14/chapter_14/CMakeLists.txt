cmake_minimum_required(VERSION 3.8)

set(PROJECT_NAME chapter_14)
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

file(GLOB SOURCES_LIB "${PROJECT_SOURCE_DIR}/src/lib/*.cpp" ${PROJECT_SOURCE_DIR}/src/main.cpp)
add_executable(${PROJECT_NAME} ${SOURCES_LIB})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

add_executable(face_detection ${PROJECT_SOURCE_DIR}/src/face_detection.cpp)
target_link_libraries(face_detection ${OpenCV_LIBS})