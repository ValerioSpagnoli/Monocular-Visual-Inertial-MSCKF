cmake_minimum_required(VERSION 3.18)
project(msckf_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The core only depends on Eigen (header-only), so it builds unchanged with the Android NDK.
# The front-end, the desktop application and the tests are desktop-only options.
option(MSCKF_BUILD_TESTS "Build the Catch2 unit tests" ON)
option(MSCKF_BUILD_APP "Build the XFeat front-end (OpenCV, ONNX Runtime) and the desktop application" ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(msckf_core STATIC
  src/imu.cpp
  src/geometry.cpp
  src/msckf.cpp
  src/pnp.cpp
  src/pose_graph.cpp
  src/loop_closure.cpp
  src/vio_system.cpp
)
target_include_directories(msckf_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(msckf_core PUBLIC Eigen3::Eigen)
target_compile_options(msckf_core PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra>)

if(MSCKF_BUILD_APP)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
  find_package(onnxruntime REQUIRED)
  add_library(msckf_frontend STATIC frontend/src/xfeat.cpp)
  target_include_directories(msckf_frontend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/frontend/include)
  target_link_libraries(msckf_frontend PUBLIC msckf_core ${OpenCV_LIBS} onnxruntime::onnxruntime)
  target_compile_options(msckf_frontend PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra>)

  # Desktop application: datasets, evaluation and Rerun visualization
  find_package(yaml-cpp REQUIRED)
  find_package(CLI11 REQUIRED)
  find_package(rerun_sdk REQUIRED)
  add_executable(msckf_run app/main.cpp app/dataset.cpp app/evaluation.cpp app/visualizer.cpp)
  target_link_libraries(msckf_run PRIVATE msckf_frontend yaml-cpp::yaml-cpp CLI11::CLI11 rerun_sdk)
  target_compile_options(msckf_run PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra>)
endif()

if(MSCKF_BUILD_TESTS)
  find_package(Catch2 3 REQUIRED)
  add_executable(msckf_tests tests/test_core.cpp tests/test_loop_closure.cpp)
  target_link_libraries(msckf_tests PRIVATE msckf_core Catch2::Catch2WithMain)
  enable_testing()
  add_test(NAME msckf_tests COMMAND msckf_tests)

  if(MSCKF_BUILD_APP)
    add_executable(msckf_app_tests tests/test_app.cpp app/dataset.cpp)
    target_link_libraries(msckf_app_tests PRIVATE msckf_frontend yaml-cpp::yaml-cpp Catch2::Catch2WithMain)
    target_compile_definitions(msckf_app_tests PRIVATE MSCKF_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
                                                       MSCKF_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../models/xfeat.onnx")
    add_test(NAME msckf_app_tests COMMAND msckf_app_tests)
  endif()
endif()
