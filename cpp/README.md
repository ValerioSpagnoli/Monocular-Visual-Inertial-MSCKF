# MSCKF C++ core

C++17 port of the filter in `src/msckf/MSCKF.py`, meant to run on Android. It depends only on Eigen (header-only),
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
```

## Build and test (desktop)

```bash
pixi run build-cpp   # core, Catch2 tests and the Python module msckf_cpp in build/cpp
pixi run test-cpp    # C++ unit tests
pixi run test        # Python tests, including tests/test_cpp_parity.py (Python vs C++)
pixi run main --source euroc --sequence V1_01_easy --backend cpp
```

`tests/test_cpp_parity.py` runs both implementations on the same synthetic visual-inertial sequence: poses and
covariance agree to about 1e-12. `benchmark.py -- --backend cpp` compares them on the datasets.

## Android

```bash
pixi run build-android                # arm64-v8a, API 24 -> build/android-arm64-v8a/libmsckf_core.a
ABI=x86_64 pixi run build-android     # emulator
```

Requires the NDK (Android Studio: Settings > Languages & Frameworks > Android SDK > SDK Tools > NDK (Side by side)),
found through `ANDROID_NDK_HOME` or in `~/Library/Android/sdk/ndk`.
