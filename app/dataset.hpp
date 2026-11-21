#pragma once

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "msckf/imu.hpp"
#include "msckf/math.hpp"

namespace msckf {

/// IMU noise densities (continuous time), as declared in the dataset calibration.
struct ImuNoise {
    double accelerometer_noise_density = 0.0;
    double gyroscope_noise_density = 0.0;
    double accelerometer_random_walk = 0.0;
    double gyroscope_random_walk = 0.0;
};

struct Image {
    double timestamp = 0.0;
    std::string path;
};

/// A visual-inertial sequence ready for the filter: IMU samples, images, calibration and the ground-truth IMU pose
/// at every IMU sample (empty where not available). Timestamps are in seconds from the first IMU sample.
struct Sequence {
    std::vector<ImuMeasurement> imu;
    std::vector<Image> images;
    std::vector<std::optional<Pose>> ground_truth;

    Mat3 K = Mat3::Identity();   // intrinsics of the (undistorted) images
    int width = 0, height = 0;
    Pose T_I_C;                  // camera pose in the IMU frame
    ImuNoise imu_noise;          // as declared by the dataset (datasheet values)
    cv::Mat undistort_map_x, undistort_map_y;  // empty if the images are already undistorted

    cv::Mat loadImage(std::size_t index) const;  // BGR, undistorted
};

/// EuRoC MAV sequence in ASL format: <root>/euroc/<name>/mav0/{cam0, imu0, state_groundtruth_estimate0}.
Sequence loadEuroc(const std::string& root, const std::string& name);

}  // namespace msckf
