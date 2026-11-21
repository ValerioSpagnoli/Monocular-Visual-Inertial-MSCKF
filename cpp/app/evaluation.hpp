#pragma once

#include <optional>
#include <string>

#include "msckf/math.hpp"

namespace msckf {

/// Accuracy and consistency of the estimate against the ground truth, accumulated at every IMU step.
/// Position errors are in the world frame and orientation errors are the local error of the filter
/// (R_true = R_est exp(δθ)), i.e. in the frames of its covariance. Real datasets have the ground truth in their own
/// world frame: it is aligned with 4 dof (position and yaw) at its first available pose.
class Evaluator {
public:
    explicit Evaluator(bool align_ground_truth) : align_(align_ground_truth) {}

    /// estimate: odometry pose; covariance_position/orientation: 3x3 blocks of the filter; corrected: pose after
    /// loop closure (optional). Returns false if there is no ground truth for this step.
    bool add(const Pose& estimate, const Mat3& covariance_position, const Mat3& covariance_orientation,
             const std::optional<Pose>& ground_truth, const std::optional<Pose>& corrected = std::nullopt);

    std::string summary() const;
    std::string markdown() const;  // for the Rerun metrics panel

    const std::optional<Pose>& alignedGroundTruth() const { return last_ground_truth_; }
    Vec3 lastPositionError() const { return last_position_error_; }

private:
    bool align_;
    std::optional<Pose> T_est_gt_;  // ground-truth world -> estimator world
    std::optional<Pose> last_ground_truth_;
    long long steps_ = 0;
    double distance_ = 0.0;
    double ate_squared_ = 0.0, aoe_squared_ = 0.0, corrected_squared_ = 0.0;
    long long corrected_steps_ = 0;
    long long inside_3sigma_ = 0;
    double nees_position_ = 0.0, nees_orientation_ = 0.0;
    Vec3 last_position_error_ = Vec3::Zero();
    Vec3 last_sigma_ = Vec3::Zero();
};

}  // namespace msckf
