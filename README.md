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

# Getting Started

## Clone
```bash
git clone --recursive git@github.com:ValerioSpagnoli/Monocular-Visual-Inertial-MSCKF.git
```

## Installation 
This project uses [Pixi](https://pixi.sh/latest/) as the package manager and virtual environment manager. Supported platforms are Linux (`linux-64`) and macOS on Apple Silicon (`osx-arm64`). To install the dependencies:
```bash
pixi install
```

Note: to install Pixi run `curl -fsSL https://pixi.sh/install.sh | bash`

## Datasets
The datasets must be downloaded into the `data/` folder. The project has been tested using synthetic and photorealistic datasets from [TartanAir](https://theairlab.org/tartanair-dataset/) and [Pering Laboratory](https://peringlab.org/lmdata/), but it can be run with any sequence of RGB images and ground truth trajectory data (needed to generate synthetic IMU data).  

### Dataset structure
The dataset structure must be as follows:

```
├── data
│   └── <source>
│       ├── camera_info.csv
│       └── <sequence>
│           ├── cam 
│           │   ├── 000000.png  # 1st image
│           │   ├── 000001.png
│           │   ├── 000002.png
│           │   ├── ...
│           │   └── 00000N.png  # Nth image
│           └── trajectory.csv
```

where:
- **source** can be, for example, `tartanair` (can be any name)
- **sequence** can be, for example, `P001` (can be any name)
- `camera_info.csv` is the file which contains the camera information of the chosen dataset and must have the following structure:
  ```
  R00,R01,R02,R10,R11,R12,R20,R21,R22,fx,fy,px,py,w,h
  0,0,1,-1,0,0,0,-1,0,320,320,320,240,640,480
  ```
  with the correct values depending on the chosen dataset.
- `trajectory.csv` is the file that contains the ground truth trajectory of the sequence. It has a different structure depending on the chosen source dataset. The currently supported datasets are:
  - **tartanair**: with the structure `'x'  'y'  'z'  'qx'  'qy'  'qz'  'qw'` (separator: space, i.e., `' '`)
  - **peringlab**: with the structure `'timestamp','x','y','z','qw','qx','qy','qz'` (separator: comma, i.e., `','`)

When the main script is run, other files will be automatically created by the `photorealistic_generator.py` module. The final structure of the data folder will be:

```
├── data
│   └── <source>
│       ├── camera_info.csv
│       └── <sequence>
│           ├── cam 
│           │   ├── 000000.png  # 1st image
│           │   ├── 000001.png
│           │   ├── 000002.png
│           │   ├── ...
│           │   └── 00000N.png  # Nth image
│           ├── camera.csv 
│           ├── imu.csv
│           ├── imu_gt.csv
│           ├── <noise_level>
│           │   ├── ATE_AOE.png
│           │   ├── recording.rrd
│           │   ├── results.txt
│           │   └── RTE_ROE.png
│           ├── trajectory.csv
│           └── transforms_gt.csv
```

where the results of the experiment will be saved in the `<noise_level>` folder.

## Run
To run the main script: 
```bash
pixi run main
```

The script accepts the following command-line parameters:
- `--source`: The name of the folder that identifies the source dataset (`tartanair`, `peringlab`, `synthetic`, `euroc`)
- `--sequence`: The name of the folder that identifies the sequence on which the experiment will be performed (e.g., `P001`, `ME007`, `V1_01_easy`)
- `--max_frames`: Maximum number of frames to process (integer, e.g., `1000`)
- `--noise_level`: The noise level of the generated IMU data (choices: `low`, `mid`, `high`)
- `--save_results`: Flag to save the results to the `<noise_level>` folder (no value needed)
- `--device`: Torch device used by XFeat (choices: `auto`, `cpu`, `mps`, `cuda`; default `auto` picks CUDA, then Apple GPU (MPS), then CPU)

Example usage:
```bash
pixi run main --source tartanair --sequence P001 --max_frames 1000 --noise_level high --save_results
pixi run main --source euroc --sequence V1_01_easy --max_frames 30000 --save_results
```

### EuRoC MAV
Download the ASL-format sequences from the [ETH Research Collection](https://www.research-collection.ethz.ch/entities/researchdata/bcaf173e-5dac-484b-bc37-faf97a594f1f) and extract them as `data/euroc/<sequence>/mav0/` (only `cam0`, `imu0` and `state_groundtruth_estimate0` are needed). IMU data is real, so `--noise_level` is ignored; calibration and IMU noise are read from the `sensor.yaml` files and the results are saved to `data/euroc/<sequence>/results/`.

## Visualization
The run is streamed to a [Rerun](https://rerun.io) viewer with a predefined layout:
- **Map**: estimated (blue) and ground-truth (green) trajectories, current body poses, camera frustum with the live image, sliding window of camera clones, triangulated landmarks (grey; those used in the current update in orange) and the 3σ position uncertainty.
- **Camera**: undistorted image with features and their tracks. Colors: orange = used in the EKF update, green = tracked, light blue = new, magenta = discarded by the χ² gating test, red = match discarded by the epipolar/homography test.
- **Metrics** and **Filter status**: running ATE/RTE/ROE, consistency w.r.t. the 3σ bounds, feature counts, estimated IMU biases and loop rate.

With `--save_results` the same stream is also written to `recording.rrd`, which can be reopened with `pixi run rerun <path>/recording.rrd`.
