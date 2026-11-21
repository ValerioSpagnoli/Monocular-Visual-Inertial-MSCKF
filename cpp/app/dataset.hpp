#pragma once

#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "msckf/imu.hpp"
#include "msckf/math.hpp"

namespace msckf {

/// IMU noise densities (continuous time).
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
    ImuNoise imu_noise;          // as declared by the dataset or used by the generator

    bool real_imu = false;              // real sensor data (EuRoC) or IMU generated from the ground truth
    bool align_ground_truth = false;    // ground truth in its own world frame: align it with 4 dof at its first pose
    cv::Mat undistort_map_x, undistort_map_y;  // empty if the images are already undistorted

    cv::Mat loadImage(std::size_t index) const;  // BGR, undistorted
};

/// EuRoC MAV sequence in ASL format: <root>/euroc/<name>/mav0/{cam0, imu0, state_groundtruth_estimate0}.
Sequence loadEuroc(const std::string& root, const std::string& name);

/// Photorealistic sequence with images and ground-truth poses only (TartanAir, PeringLab): the IMU is generated
/// from a smooth interpolation of the trajectory (PCHIP positions, slerp rotations) so that the filter integration
/// reproduces it exactly, plus white noise and bias random walks with the given densities.
/// TartanAir poses are in the NED camera frame, used as IMU frame; PeringLab poses are camera poses and
/// camera_info.csv gives the camera rotation in the IMU frame.
Sequence loadPhotorealistic(const std::string& root, const std::string& source, const std::string& name, const ImuNoise& noise,
                            unsigned seed = 42);

/// Monotone piecewise cubic Hermite interpolation (as scipy.interpolate.PchipInterpolator) of each column of `values`.
class Pchip {
public:
    Pchip(const std::vector<double>& times, const std::vector<Vec3>& values);
    Vec3 value(double t) const;
    Vec3 derivative(double t) const;

private:
    std::size_t segment(double t) const;
    std::vector<double> times_;
    std::vector<Vec3> values_, slopes_;
};

}  // namespace msckf
