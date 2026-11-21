#include "evaluation.hpp"

#include <cstdio>

#include "msckf/pose_graph.hpp"

namespace {
constexpr double kDegrees = 180.0 / 3.14159265358979323846;
}

namespace msckf {

bool Evaluator::add(const Pose& estimate, const Mat3& covariance_position, const Mat3& covariance_orientation,
                    const std::optional<Pose>& ground_truth_raw, const std::optional<Pose>& corrected) {
    if (!ground_truth_raw) return false;
    if (align_ && !T_est_gt_) {
        // Roll and pitch are observable, so only yaw (about gravity, z in both frames) and position are aligned
        const Mat3 R_yaw = rotationZ(yawOf(estimate.R * ground_truth_raw->R.transpose()));
        T_est_gt_ = Pose(R_yaw, estimate.t - R_yaw * ground_truth_raw->t);
    }
    const Pose ground_truth = T_est_gt_ ? *T_est_gt_ * *ground_truth_raw : *ground_truth_raw;
    if (last_ground_truth_) distance_ += (ground_truth.t - last_ground_truth_->t).norm();
    last_ground_truth_ = ground_truth;

    const Vec3 position_error = estimate.t - ground_truth.t;
    const Vec3 orientation_error = logSO3(estimate.R.transpose() * ground_truth.R);
    const Vec3 sigma = covariance_position.diagonal().cwiseMax(0.0).cwiseSqrt();

    ++steps_;
    ate_squared_ += position_error.squaredNorm();
    aoe_squared_ += orientation_error.squaredNorm();
    inside_3sigma_ += (position_error.cwiseAbs().array() <= 3.0 * sigma.array()).all() ? 1 : 0;
    nees_position_ += position_error.dot(covariance_position.completeOrthogonalDecomposition().pseudoInverse() * position_error);
    nees_orientation_ += orientation_error.dot(covariance_orientation.completeOrthogonalDecomposition().pseudoInverse() * orientation_error);
    if (corrected) {
        corrected_squared_ += (corrected->t - ground_truth.t).squaredNorm();
        ++corrected_steps_;
    }
    last_position_error_ = position_error;
    last_sigma_ = sigma;
    return true;
}

std::string Evaluator::summary() const {
    const double n = static_cast<double>(std::max<long long>(steps_, 1));
    char buffer[1024];
    std::snprintf(buffer, sizeof(buffer),
                  " - ATE (evaluated over %lld steps, %.2f m):\n"
                  "   - Position RMSE:  %.4f m\n"
                  "   - Orient. RMSE:   %.3f deg\n"
                  "   - Inside 3σ:      %.1f %% of steps\n"
                  "   - Mean NEES:      position %.2f, orientation %.2f (consistent ≈ 3)\n",
                  steps_, distance_, std::sqrt(ate_squared_ / n), std::sqrt(aoe_squared_ / n) * kDegrees,
                  100.0 * inside_3sigma_ / n, nees_position_ / n, nees_orientation_ / n);
    std::string text = buffer;
    if (corrected_steps_ > 0) {
        std::snprintf(buffer, sizeof(buffer), "   - Loop-corrected position RMSE: %.4f m\n", std::sqrt(corrected_squared_ / corrected_steps_));
        text += buffer;
    }
    return text;
}

std::string Evaluator::markdown() const {
    const double n = static_cast<double>(std::max<long long>(steps_, 1));
    const double ate = std::sqrt(ate_squared_ / n);
    char buffer[2048];
    std::snprintf(buffer, sizeof(buffer),
                  "| Metric | Value |\n|---|---|\n"
                  "| ATE position RMSE | %.3f m |\n"
                  "| ATE position (current) | %.3f m |\n"
                  "| ATE orientation RMSE | %.2f deg |\n"
                  "| Drift (ATE RMSE / distance) | %.2f %% |\n"
                  "| Distance traveled (gt) | %.2f m |\n"
                  "| Position error inside 3σ | %.1f %% of steps |\n"
                  "| Mean NEES position / orientation (consistent ≈ 3) | %.1f / %.1f |\n"
                  "| Current error x / y / z (world) | %+.3f / %+.3f / %+.3f m |\n"
                  "| Current 3σ x / y / z | %.3f / %.3f / %.3f m |\n",
                  ate, last_position_error_.norm(), std::sqrt(aoe_squared_ / n) * kDegrees, 100.0 * ate / std::max(distance_, 1e-9),
                  distance_, 100.0 * inside_3sigma_ / n, nees_position_ / n, nees_orientation_ / n, last_position_error_.x(),
                  last_position_error_.y(), last_position_error_.z(), 3 * last_sigma_.x(), 3 * last_sigma_.y(), 3 * last_sigma_.z());
    std::string text = buffer;
    if (corrected_steps_ > 0) {
        std::snprintf(buffer, sizeof(buffer), "| Loop-corrected position RMSE | %.3f m |\n", std::sqrt(corrected_squared_ / corrected_steps_));
        text += buffer;
    }
    return text;
}

}  // namespace msckf
