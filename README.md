# Visual-Inertial MSCKF: Multi-State Constraint Kalman Filter for Monocular Visual-Inertial Navigation

This repository contains the project based on my Master's Thesis: [**Multi-State Constraint Kalman Filter for Monocular Visual-Inertial Navigation**](https://github.com/ValerioSpagnoli/Visual-Inertial-MSCKF/blob/main/thesis.pdf).

<p align="center">
  <img src="https://github.com/user-attachments/assets/3a696c51-a02f-498d-bc86-25d39afb19bc" alt="deer_running">
</p>

## Overview
This project implements and evaluates a Multi-State Constraint Kalman Filter (MSCKF) for monocular Visual-Inertial Odometry (VIO). The core idea is to fuse visual information from a single camera with measurements from an Inertial Measurement Unit (IMU) to achieve precise and computationally efficient real-time pose estimation. The system provides a robust localization solution with minimal computational overhead, making it suitable for various applications such as robotic navigation, search and rescue, and augmented reality.   

## Thesis Idea
The thesis focuses on implementing the MSCKF algorithm, which leverages an Extended Kalman Filter (EKF) framework to integrate visual observations and inertial measurements. Unlike traditional filtering approaches that include 3D feature positions directly in the state vector, the MSCKF formulates constraints between multiple camera poses that have observed the same static features. This allows the system to benefit from the information provided by these features without increasing the state vector size with every new feature, thus ensuring computational efficiency. The map is indirectly refined by optimizing camera poses within the EKF state.

Key enhancements to the standard MSCKF approach in this work include:
* **XFeat Integration**: The [XFeat](https://github.com/verlab/accelerated_features) library, a Convolutional Neural Network (CNN)-based architecture, is used for feature extraction and matching. XFeat offers a fast, precise, and hardware-independent solution for identifying and matching local features across image frames.
* **Epipolar Matching Refinement**: An epipolar geometry-based refinement step is incorporated into the matching pipeline. This significantly improves the accuracy of data association by eliminating false positive matches without a substantial increase in computational cost.

The theoretical underpinnings of the project include projective geometry, IMU modeling, and Bayesian filtering techniques, particularly the EKF.

## Results

The implemented MSCKF system was evaluated on multiple photorealistic datasets using synthetically generated IMU data derived from ground truth trajectories. The system demonstrated consistency and good accuracy even under challenging conditions such as reflections, motion blur, and varying levels of IMU noise.

Key findings from the experiments include:
* **Translational Relative Pose Error (RPE)**: Generally remained below 4% for low and medium IMU noise levels across different sequences. It increased at high noise levels, especially in visually challenging sequences with repetitive patterns.
* **Rotational Relative Pose Error (RPE)**: Showed significantly lower errors, with a maximum of only 0.7% across all sequences. This is attributed to the inherently lower noise in gyroscopes and the use of **Inverse Depth Parametrization (IDP)** for features, which allows for accurate orientation information extraction even from features with low parallax.
* **Consistency**: Absolute Trajectory Error (ATE) analysis confirmed that the estimation errors remained within the 3-sigma bounds, indicating filter consistency. However, as a VIO system without loop closure or global map optimization, drift over time is expected.
* **Real-Time Performance**: Despite being implemented in Python (which introduces computational overhead compared to C++), performance profiling on standard laptop hardware indicated that the system achieves real-time operation. The system maintained an average loop frequency of approximately 45 Hz overall. Loops relying solely on IMU measurements ran at about 171 Hz, while loops incorporating camera measurements (which are more computationally intensive) ran at around 7 Hz.

These results highlight the system's potential for precise localization in resource-constrained environments, effectively balancing computational efficiency with estimation accuracy.   

# Implementation

The original Python implementation of the thesis has been ported to **C++17** (it is still available on the `main` branch and in the history).

```
backend/                 filter: MSCKF, IMU, geometry, loop closure, VioSystem (Eigen only, builds for Android)
  include/msckf/  src/  tests/
frontend/                feature extraction: XFeat with ONNX Runtime (OpenCV, ONNX Runtime)
  include/msckf_frontend/  src/  tests/
app/                     msckf_run: EuRoC datasets, evaluation, Rerun visualization (desktop only)
models/xfeat.onnx        XFeat network
tools/                   benchmark.sh, build_android.sh, export_xfeat_onnx.py
third_party/             XFeat repository (submodule, only used to export the model)
```

| Part | Dependencies | Contents |
|---|---|---|
| Backend | Eigen | MSCKF (IMU propagation with observability constraint, descriptor matching with epipolar checks, Gauss-Newton triangulation, null-space update, χ² gating, zero-velocity updates), loop closure with a 4-dof pose graph, `VioSystem` entry point. |
| Frontend | OpenCV, ONNX Runtime | Keypoints and descriptors behind a common `FeatureExtractor`: XFeat from `models/xfeat.onnx` (reproducing `XFeat.detectAndCompute`), LiftFeat with the same interface, ORB, SIFT and AKAZE from OpenCV. |
| App | yaml-cpp, CLI11, Rerun C++ SDK | `msckf_run`: EuRoC sequences, evaluation, visualization. |

See [backend/README.md](backend/README.md) for the backend API and the Android build.

# Getting Started

## Clone
```bash
git clone --recursive git@github.com:ValerioSpagnoli/Monocular-Visual-Inertial-MSCKF.git
```

## Installation
This project uses [Pixi](https://pixi.sh/latest/) for the toolchain and all the dependencies (Linux `linux-64` and macOS on Apple Silicon `osx-arm64`):
```bash
pixi run build    # build everything in build/
pixi run test     # unit tests
```

Note: to install Pixi run `curl -fsSL https://pixi.sh/install.sh | bash`

## Datasets
The datasets must be placed in the `data/` folder.

### EuRoC MAV (real IMU)
Download the ASL-format sequences from the [ETH Research Collection](https://www.research-collection.ethz.ch/entities/researchdata/bcaf173e-5dac-484b-bc37-faf97a594f1f) and extract them as `data/euroc/<sequence>/mav0/` (only `cam0`, `imu0` and `state_groundtruth_estimate0` are needed). Calibration and IMU noise are read from the `sensor.yaml` files; the images are undistorted so that the filter uses a pinhole model.

## Run
```bash
pixi run main --sequence V1_01_easy
pixi run main --sequence V1_02_medium --loop_closure --save_results
pixi run main --sequence V1_03_difficult --no_viewer
```
Main options (`pixi run main --help` for all of them):
- `--sequence`, `--max_frames` (IMU samples), `--data` (dataset root, default `./data`)
- `--loop_closure`: keyframe-based loop closure; the summary also reports the loop-corrected ATE
- `--save_results`: `results.txt`, `trajectory.csv` and the Rerun recording in `data/euroc/<sequence>/results`
- `--no_viewer`: do not stream to Rerun
- `--coreml`: run XFeat with CoreML on Apple hardware
- Features: `--features xfeat|liftfeat|orb|sift|akaze` (default `xfeat`), `--top_k` (300), `--min_cosine_similarity` (default per features). `models/liftfeat.onnx` is generated locally with `pixi run -e export export-liftfeat` (LiftFeat has no license, so it is not committed).
- Tuning: `--sigma_image`, `--imu_noise_scale` (default ×10 on the datasheet white noise), `--imu_walk_scale`, `--epipolar_threshold`, `--max_triangulation_error`, `--zupt_max_disparity`, `--min_parallax`, `--max_track_length`

### Benchmark
```bash
pixi run benchmark                  # EuRoC V1_01-03, markdown table
pixi run benchmark --loop_closure   # extra options are forwarded to msckf_run
```

## Visualization
The run is streamed to a [Rerun](https://rerun.io) viewer: estimated (blue), ground-truth (green) and loop-corrected (orange) trajectories, body poses, the camera frustum with the live image, the sliding window of clones, the triangulated landmarks and the 3σ position uncertainty; the camera image shows the features and their tracks (orange = used in the EKF update, green = tracked, light blue = new, magenta = discarded by the χ² gating, red = match rejected by the epipolar/homography test); two text panels report the running metrics and the filter status.

## Regenerating the XFeat model
`models/xfeat.onnx` is exported from the [XFeat](https://github.com/verlab/accelerated_features) submodule with PyTorch, in a separate environment:
```bash
pixi run -e export export-xfeat
```
