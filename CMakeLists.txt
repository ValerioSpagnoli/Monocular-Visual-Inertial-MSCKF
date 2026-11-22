cmake_minimum_required(VERSION 3.18)
project(msckf VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# src/core/  library msckf_core: filter (MSCKF, IMU, geometry, loop closure; Eigen only, builds unchanged with the
#            Android NDK) and feature extraction (XFeat, LiftFeat, ORB, SIFT, AKAZE; OpenCV and ONNX Runtime)
# src/app/   desktop application msckf_run (EuRoC, evaluation, Rerun): desktop only
option(MSCKF_BUILD_FEATURES "Build the feature extractors in the core (OpenCV, ONNX Runtime)" ON)
option(MSCKF_BUILD_APP "Build the desktop application msckf_run (requires the feature extractors)" ON)
option(MSCKF_BUILD_TESTS "Build the Catch2 unit tests" ON)

set(MSCKF_WARNINGS $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra>)
set(CORE ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
set(APP ${CMAKE_CURRENT_SOURCE_DIR}/src/app)

#* Core
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
add_library(msckf_core STATIC
  ${CORE}/src/imu.cpp
  ${CORE}/src/geometry.cpp
  ${CORE}/src/msckf.cpp
  ${CORE}/src/pnp.cpp
  ${CORE}/src/pose_graph.cpp
  ${CORE}/src/loop_closure.cpp
  ${CORE}/src/vio_system.cpp
)
target_include_directories(msckf_core PUBLIC ${CORE}/include)
target_link_libraries(msckf_core PUBLIC Eigen3::Eigen)
target_compile_options(msckf_core PRIVATE ${MSCKF_WARNINGS})

if(MSCKF_BUILD_FEATURES)
  # ORB, SIFT and AKAZE are in features2d up to OpenCV 4 (e.g. the Android SDK); in OpenCV 5 AKAZE is in xfeatures2d
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
  if(OpenCV_VERSION VERSION_GREATER_EQUAL 5)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs features xfeatures2d)
  else()
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs features2d)
  endif()
  find_package(onnxruntime REQUIRED)
  target_sources(msckf_core PRIVATE ${CORE}/src/xfeat.cpp ${CORE}/src/opencv_features.cpp)
  target_link_libraries(msckf_core PUBLIC ${OpenCV_LIBS} onnxruntime::onnxruntime)
endif()

#* Desktop application
if(MSCKF_BUILD_APP AND MSCKF_BUILD_FEATURES)
  find_package(yaml-cpp REQUIRED)
  find_package(CLI11 REQUIRED)
  find_package(rerun_sdk REQUIRED)
  add_executable(msckf_run ${APP}/src/main.cpp ${APP}/src/dataset.cpp ${APP}/src/evaluation.cpp ${APP}/src/visualizer.cpp)
  target_include_directories(msckf_run PRIVATE ${APP}/include)
  target_link_libraries(msckf_run PRIVATE msckf_core yaml-cpp::yaml-cpp CLI11::CLI11 rerun_sdk)
  target_compile_options(msckf_run PRIVATE ${MSCKF_WARNINGS})
endif()

#* Tests
if(MSCKF_BUILD_TESTS)
  find_package(Catch2 3 REQUIRED)
  enable_testing()
  set(TEST_SOURCES ${CORE}/tests/test_core.cpp ${CORE}/tests/test_loop_closure.cpp)
  if(MSCKF_BUILD_FEATURES)
    list(APPEND TEST_SOURCES ${CORE}/tests/test_xfeat.cpp ${CORE}/tests/test_opencv_features.cpp)
  endif()
  add_executable(msckf_tests ${TEST_SOURCES})
  target_link_libraries(msckf_tests PRIVATE msckf_core Catch2::Catch2WithMain)
  target_compile_definitions(msckf_tests PRIVATE MSCKF_TEST_DATA_DIR="${CORE}/tests/data"
                                                 MSCKF_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/xfeat.onnx")
  add_test(NAME msckf_tests COMMAND msckf_tests)
endif()
