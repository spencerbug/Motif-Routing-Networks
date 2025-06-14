cmake_minimum_required(VERSION 3.12)
project(dwt3d_cpp LANGUAGES CXX)

# Use FetchContent to get pybind11 if not available
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# Set the source file
set(SOURCE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/dwt3d_cpp.cpp)

# Create the Python module
pybind11_add_module(dwt3d_cpp ${SOURCE_FILE})

# Set C++ standard
set_target_properties(dwt3d_cpp PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)
