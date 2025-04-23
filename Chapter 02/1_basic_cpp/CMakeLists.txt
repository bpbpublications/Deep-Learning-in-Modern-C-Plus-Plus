cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(hello_world
        VERSION 1.0
        DESCRIPTION "Hello World application"
        LANGUAGES CXX)
        
# Default to Release build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# C++17 is mandatory
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/hello_world.cpp)

# Compiler Options: will enable almost all of the warnings available for a compiler
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)

# defining an output folder
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "bin")
