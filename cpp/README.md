# MSCKF C++ core

C++17 implementation of the filter (ported from the Python code of the thesis), meant to run on Android. It depends only on Eigen (header-only),
so the same sources build on desktop and with the Android NDK.

The core covers everything after feature extraction: IMU propagation with the observability constraint, camera
clones, descriptor matching and epipolar/homography checks, Gauss-Newton triangulation, the null-space MSCKF update
with chi-square gating, zero-velocity updates and clone pruning. Feature extraction (XFeat) stays outside: on the
desktop it runs in PyTorch, on the phone it will run in an on-device runtime (e.g. ONNX Runtime or TFLite) and pass
keypoints, descriptors and scores to the core.

## API

```cpp
#include "msckf/msckf.hpp"

msckf::Parameters parameters;          // intrinsics K of the undistorted image, camera pose in the IMU frame,
parameters.K = K;                      // IMU noise densities, thresholds (same meaning as in MSCKFParameters)
parameters.T_I_C = msckf::Pose(R_I_C, t_I_C);
msckf::Msckf filter(parameters);

filter.addImu({timestamp, gyro, accel});          // every IMU sample, accel = specific force
filter.addImage({keypoints, descriptors, scores}); // after the IMU sample at the image timestamp

filter.imu().T_W_I;   // current pose, also v_W_I, biases
filter.covariance();  // [δθ, δbg, δv, δba, δp] of the IMU, then [δθ, δp] of each clone

// With loop closure on keyframes:
msckf::VioSystem system(parameters, msckf::LoopClosureParameters{}, /*enable_loop_closure=*/true);
system.addImu(...); system.addImage(timestamp, features);
system.correctedPose();  // odometry pose corrected with the 4-dof pose graph
```

## Build and test (desktop)

```bash
pixi run build-cpp   # core, front-end, msckf_run and the Catch2 tests in build/cpp
pixi run test-cpp    # core tests (filter, geometry, loop closure) and application tests (XFeat vs PyTorch reference, datasets)
pixi run main --source euroc --sequence V1_01_easy
```

The front-end (`cpp/frontend`, OpenCV and ONNX Runtime) and the application (`cpp/app`) are enabled by
`MSCKF_BUILD_APP` (default ON on desktop).

## Android

```bash
pixi run build-android                # arm64-v8a, API 24 -> build/android-arm64-v8a/libmsckf_core.a
ABI=x86_64 pixi run build-android     # emulator
```

Requires the NDK (Android Studio: Settings > Languages & Frameworks > Android SDK > SDK Tools > NDK (Side by side)),
found through `ANDROID_NDK_HOME` or in `~/Library/Android/sdk/ndk`.
