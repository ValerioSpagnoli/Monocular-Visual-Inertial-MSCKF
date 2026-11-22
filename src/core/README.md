# Core

Library `msckf_core`: the C++17 filter (ported from the Python code of the thesis), meant to run on Android, and the
feature extractors. The filter depends only on Eigen (header-only), so the same sources build on desktop and with the
Android NDK; the extractors (`xfeat`, `opencv_features`) need OpenCV and ONNX Runtime and are built when
`MSCKF_BUILD_FEATURES` is ON.

The filter covers everything after feature extraction: static initialization, IMU propagation with the observability
constraint, camera clones, descriptor matching and epipolar/homography checks, Gauss-Newton triangulation, the
null-space MSCKF update with chi-square gating, zero-velocity updates, clone pruning and loop closure. It takes
keypoints, descriptors and scores from any `FeatureExtractor` (XFeat with ONNX Runtime, also available on Android,
LiftFeat, ORB, SIFT, AKAZE).

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
pixi run build   # msckf_core, msckf_run and the Catch2 tests (msckf_tests) in build/
pixi run test    # filter, geometry, loop closure, feature extractors (XFeat vs PyTorch reference)
pixi run main --sequence V1_01_easy
```

The feature extractors (OpenCV and ONNX Runtime) and the application (`src/app/`) are enabled by
`MSCKF_BUILD_FEATURES` and `MSCKF_BUILD_APP` (both ON by default).

## Android

```bash
pixi run build-android                # arm64-v8a, API 24 -> build/android-arm64-v8a/libmsckf_core.a (filter only)
ABI=x86_64 pixi run build-android     # emulator
```

Requires the NDK (Android Studio: Settings > Languages & Frameworks > Android SDK > SDK Tools > NDK (Side by side)),
found through `ANDROID_NDK_HOME` or in `~/Library/Android/sdk/ndk`.
