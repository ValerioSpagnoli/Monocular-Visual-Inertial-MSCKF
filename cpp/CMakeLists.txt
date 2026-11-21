cmake_minimum_required(VERSION 3.18)
project(msckf_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The core only depends on Eigen (header-only), so it builds unchanged with the Android NDK.
# Python bindings and tests are desktop-only options.
option(MSCKF_BUILD_PYTHON "Build the pybind11 module msckf_cpp" ON)
option(MSCKF_BUILD_TESTS "Build the Catch2 unit tests" ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(msckf_core STATIC
  src/imu.cpp
  src/geometry.cpp
  src/msckf.cpp
)
target_include_directories(msckf_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(msckf_core PUBLIC Eigen3::Eigen)
target_compile_options(msckf_core PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra>)

if(MSCKF_BUILD_PYTHON)
  find_package(Python 3 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(msckf_cpp python/bindings.cpp)
  target_link_libraries(msckf_cpp PRIVATE msckf_core)
endif()

if(MSCKF_BUILD_TESTS)
  find_package(Catch2 3 REQUIRED)
  add_executable(msckf_tests tests/test_core.cpp)
  target_link_libraries(msckf_tests PRIVATE msckf_core Catch2::Catch2WithMain)
  enable_testing()
  add_test(NAME msckf_tests COMMAND msckf_tests)
endif()
