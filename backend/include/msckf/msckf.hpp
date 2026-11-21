#pragma once

#include <map>
#include <vector>

#include "msckf/geometry.hpp"
#include "msckf/imu.hpp"
#include "msckf/loop_closure.hpp"
#include "msckf/math.hpp"

namespace msckf {

struct Parameters {
    // Camera
    Mat3 K = Mat3::Identity();   // pinhole intrinsics of the (undistorted) image
    Pose T_I_C;                  // camera pose in the IMU frame
    double sigma_image = 2.0;    // keypoint noise [px]

    // IMU (continuous-time densities)
    bool only_imu = false;
    double accelerometer_noise_density = 1e-3;
    double accelerometer_random_walk = 1e-5;
    double gyroscope_noise_density = 1e-4;
    double gyroscope_random_walk = 1e-6;
    Vec3 gravity{0.0, 0.0, -9.81};
    bool estimate_initial_gyroscope_bias = false;
    // Add the uncertainty of the gravity direction estimated from the static samples to the initial orientation
    // covariance (C++ only; false reproduces the Python implementation)
    bool initial_tilt_covariance_from_noise = true;
    Eigen::Matrix<double, 15, 1> initial_covariance_diagonal = Eigen::Matrix<double, 15, 1>::Zero();  // [δθ, δbg, δv, δba, δp]

    // Features (tracked by the frontend)
    bool use_parallax = true;
    double min_parallax = 20.0;                    // [deg]
    int min_number_of_frames_to_be_lost = 1;
    int min_number_of_frames_to_be_tracked = 5;
    int max_number_of_camera_states = 30;
    double max_triangulation_error = 6.0;          // maximum RMS reprojection error of a triangulated feature [px]

    // Zero-velocity update: when the tracked features do not move, the platform is static and v = 0 is a measurement
    double zupt_max_disparity = 1.0;               // median feature displacement between consecutive images [px], <= 0 disables it
    double zupt_sigma_velocity = 0.01;             // [m/s]

    // Update strategy (C++ only; set max_track_length = 0 and update_features_on_prune = false for the behaviour of
    // the Python implementation). Features are always used when lost or with min_parallax.
    int max_track_length = 10;                     // also use still-tracked features with this many observations, 0 disables it
    bool update_features_on_prune = true;          // use every feature seen by the pruned clones before removing them
};

/// Observation of a tracked feature in one image. The frontend associates the observations of the same feature over
/// time through its id; the descriptor is only kept for the loop closure.
struct FeatureObservation {
    long long id = 0;
    Vec2 keypoint = Vec2::Zero();  // [px], in the undistorted image
    Eigen::VectorXf descriptor;
    float score = 1.0f;
};

/// Output of the frontend for one image.
struct TrackedFrame {
    std::vector<FeatureObservation> observations;
};

struct CameraClone {
    Pose T_W_C;
    Pose T_W_C_null;  // first estimate, used by the observability constraint
};

struct Feature {
    std::vector<Vec2> keypoints;
    std::vector<Eigen::VectorXf> descriptors;
    std::vector<float> scores;
    std::vector<long long> camera_ids;
    std::vector<Line> lines;
    Vec3 point = Vec3::Zero();  // last triangulated world point
    int tracked_for_n_frames = 0;
    int lost_for_n_frames = 0;
};

/// Multi-State Constraint Kalman Filter for monocular visual-inertial odometry (backend: features are tracked by the
/// frontend, a feature not observed in an image counts as lost for that image).
/// State: [δθ, δbg, δv, δba, δp] of the IMU followed by [δθ, δp] of each camera clone (local orientation error,
/// R_true = R_est exp(δθ)).
class Msckf {
public:
    explicit Msckf(const Parameters& parameters);

    void addImu(const ImuMeasurement& measurement);
    void addImage(const TrackedFrame& frame);

    // State
    const ImuState& imu() const { return imu_; }
    const MatX& covariance() const { return covariance_; }
    const std::map<long long, CameraClone>& cameras() const { return cameras_; }
    const std::map<long long, Feature>& features() const { return features_; }

    // Diagnostics of the last image
    const std::vector<long long>& updatedFeatureIds() const { return updated_feature_ids_; }
    const std::vector<long long>& gatedFeatureIds() const { return gated_feature_ids_; }
    const std::map<long long, Vec3>& triangulatedPoints() const { return triangulated_points_; }

    /// Keypoints, descriptors and world points of the features observed in the last image. Points come from the last
    /// triangulation of the filter or, for tracks with at least 3 observations and enough parallax not yet used in an
    /// update, from a new triangulation on the current clone poses (only returned, the filter state is unchanged).
    KeyframeObservations currentObservations(double min_parallax_degrees = 2.0) const;

    // Counters
    long long gatingRejections() const { return gating_rejections_; }
    long long triangulationRejections() const { return triangulation_rejections_; }
    long long zeroVelocityUpdates() const { return zero_velocity_updates_; }

private:
    void processImu(const ImuMeasurement& measurement);
    void augmentState();
    void addCameraMeasurements(const TrackedFrame& frame);
    void zeroVelocityUpdate();
    void processFeatures();
    void getValidFeatures(const std::vector<long long>& candidates, std::vector<long long>& valid, std::vector<long long>& lost,
                          bool use_all_tracked = false);
    bool residualAndJacobian(const Feature& feature, VecX& r_o, MatX& H_o) const;
    bool gatingTest(const VecX& r, const MatX& H) const;
    void update(const std::vector<long long>& feature_ids);
    void correct(const MatX& K, const MatX& T_H, const MatX& R_n, const VecX& delta_x);
    void discardUsedObservations(const std::vector<long long>& feature_ids);
    void pruneCameraStates();
    void removeFeatures(const std::vector<long long>& feature_ids);
    void removeCameras(const std::vector<long long>& camera_ids);
    int cameraIndex(long long camera_id) const;

    Parameters parameters_;
    Mat3 invK_;
    double sigma_normalized_;          // keypoint noise in normalized coordinates
    double max_triangulation_error_;   // RMS reprojection error, normalized coordinates
    Eigen::Matrix<double, 12, 12> Qc_ = Eigen::Matrix<double, 12, 12>::Zero();

    ImuState imu_;
    MatX covariance_;
    std::map<long long, CameraClone> cameras_;
    std::map<long long, Feature> features_;
    std::vector<ImuMeasurement> imu_buffer_;
    bool first_image_arrived_ = false;

    std::vector<long long> updated_feature_ids_;
    std::vector<long long> gated_feature_ids_;
    std::map<long long, Vec3> triangulated_points_;
    long long gating_rejections_ = 0;
    long long triangulation_rejections_ = 0;
    long long zero_velocity_updates_ = 0;
    double last_disparity_ = -1.0;  // median displacement of the features matched in the last image [px], < 0 if none
};

}  // namespace msckf
