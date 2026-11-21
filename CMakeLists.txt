cmake_minimum_required(VERSION 3.18)
project(msckf VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# backend/   filter (MSCKF, IMU, geometry, loop closure): Eigen only, builds unchanged with the Android NDK
# frontend/  feature extraction (XFeat, LiftFeat, ORB, SIFT, AKAZE): OpenCV and ONNX Runtime, desktop and Android
# app/       desktop application msckf_run (EuRoC, evaluation, Rerun): desktop only
option(MSCKF_BUILD_FRONTEND "Build the XFeat front-end (OpenCV, ONNX Runtime)" ON)
option(MSCKF_BUILD_APP "Build the desktop application msckf_run (requires the front-end)" ON)
option(MSCKF_BUILD_TESTS "Build the Catch2 unit tests" ON)

set(MSCKF_WARNINGS $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra>)

#* Backend
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
add_library(msckf_backend STATIC
  backend/src/imu.cpp
  backend/src/geometry.cpp
  backend/src/msckf.cpp
  backend/src/pnp.cpp
  backend/src/pose_graph.cpp
  backend/src/loop_closure.cpp
  backend/src/vio_system.cpp
)
target_include_directories(msckf_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/backend/include)
target_link_libraries(msckf_backend PUBLIC Eigen3::Eigen)
target_compile_options(msckf_backend PRIVATE ${MSCKF_WARNINGS})

#* Frontend
if(MSCKF_BUILD_FRONTEND)
  # ORB, SIFT and AKAZE are in features2d up to OpenCV 4 (e.g. the Android SDK); in OpenCV 5 AKAZE is in xfeatures2d
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
  if(OpenCV_VERSION VERSION_GREATER_EQUAL 5)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs features xfeatures2d)
  else()
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs features2d)
  endif()
  find_package(onnxruntime REQUIRED)
  add_library(msckf_frontend STATIC frontend/src/xfeat.cpp frontend/src/opencv_features.cpp)
  target_include_directories(msckf_frontend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/frontend/include)
  target_link_libraries(msckf_frontend PUBLIC msckf_backend ${OpenCV_LIBS} onnxruntime::onnxruntime)
  target_compile_options(msckf_frontend PRIVATE ${MSCKF_WARNINGS})
endif()

#* Desktop application
if(MSCKF_BUILD_APP AND MSCKF_BUILD_FRONTEND)
  find_package(yaml-cpp REQUIRED)
  find_package(CLI11 REQUIRED)
  find_package(rerun_sdk REQUIRED)
  add_executable(msckf_run app/main.cpp app/dataset.cpp app/evaluation.cpp app/visualizer.cpp)
  target_link_libraries(msckf_run PRIVATE msckf_frontend yaml-cpp::yaml-cpp CLI11::CLI11 rerun_sdk)
  target_compile_options(msckf_run PRIVATE ${MSCKF_WARNINGS})
endif()

#* Tests
if(MSCKF_BUILD_TESTS)
  find_package(Catch2 3 REQUIRED)
  enable_testing()

  add_executable(msckf_backend_tests backend/tests/test_core.cpp backend/tests/test_loop_closure.cpp)
  target_link_libraries(msckf_backend_tests PRIVATE msckf_backend Catch2::Catch2WithMain)
  add_test(NAME msckf_backend_tests COMMAND msckf_backend_tests)

  if(MSCKF_BUILD_FRONTEND)
    add_executable(msckf_frontend_tests frontend/tests/test_xfeat.cpp frontend/tests/test_opencv_features.cpp)
    target_link_libraries(msckf_frontend_tests PRIVATE msckf_frontend Catch2::Catch2WithMain)
    target_compile_definitions(msckf_frontend_tests PRIVATE MSCKF_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/frontend/tests/data"
                                                            MSCKF_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/xfeat.onnx")
    add_test(NAME msckf_frontend_tests COMMAND msckf_frontend_tests)
  endif()
endif()
