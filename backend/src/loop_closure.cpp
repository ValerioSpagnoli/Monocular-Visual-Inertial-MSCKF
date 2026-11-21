#include "msckf/loop_closure.hpp"

#include <algorithm>

#include "msckf/pnp.hpp"

namespace msckf {

LoopClosure::LoopClosure(const LoopClosureParameters& parameters, const Mat3& K, const Pose& T_I_C)
    : parameters_(parameters), invK_(K.inverse()), focal_(0.5 * (K(0, 0) + K(1, 1))), T_I_C_(T_I_C) {}

bool LoopClosure::needsKeyframe(const Pose& T_W_I) const {
    if (keyframes_.empty()) return true;
    const Pose& last = keyframes_.back().T_W_I_vio;
    const double distance = (T_W_I.t - last.t).norm();
    const double angle = logSO3(last.R.transpose() * T_W_I.R).norm() * 180.0 / EIGEN_PI;
    return distance > parameters_.keyframe_distance || angle > parameters_.keyframe_angle;
}

std::optional<LoopDetection> LoopClosure::addKeyframe(double timestamp, const Pose& T_W_I, const KeyframeObservations& observations) {
    Keyframe keyframe;
    keyframe.timestamp = timestamp;
    keyframe.T_W_I_vio = T_W_I;
    const Pose T_W_C = T_W_I * T_I_C_;
    const Pose T_C_W = T_W_C.inverse();

    const Eigen::Index n = static_cast<Eigen::Index>(observations.keypoints.size());
    keyframe.descriptors = observations.descriptors;
    keyframe.descriptors.rowwise().normalize();
    keyframe.global_descriptor = n > 0 ? Eigen::VectorXf(keyframe.descriptors.colwise().mean().transpose().normalized()) : Eigen::VectorXf();
    for (Eigen::Index k = 0; k < n; ++k) {
        const Vec2& u = observations.keypoints[static_cast<std::size_t>(k)];
        keyframe.observations.push_back((invK_ * Vec3(u.x(), u.y(), 1.0)).head<2>());
        const auto& point = observations.points[static_cast<std::size_t>(k)];
        if (point) {
            const Vec3 point_camera = T_C_W.transform(*point);
            if (point_camera.z() > 0.1) {
                keyframe.landmark_rows.push_back(static_cast<int>(k));
                keyframe.landmarks_camera.push_back(point_camera);
            }
        }
    }

    // Graph node with an odometry edge from the previous keyframe. The node starts from the odometry pose with the drift
    // correction estimated so far: otherwise, until the next loop triggers an optimization, the last node would equal
    // the odometry and the correction would drop to zero. The odometry edge is unaffected (the correction is rigid).
    const int index = graph_.addNode(correction_ * T_W_I);
    if (index > 0) {
        const Pose T_prev_curr = keyframes_.back().T_W_I_vio.inverse() * T_W_I;
        graph_.addEdge(index - 1, index, T_prev_curr, parameters_.odometry_sigma_p, parameters_.odometry_sigma_yaw, false);
    }
    statistics_.mean_landmarks += (static_cast<double>(keyframe.landmark_rows.size()) - statistics_.mean_landmarks) / (keyframes_.size() + 1);
    keyframes_.push_back(std::move(keyframe));

    std::optional<LoopDetection> loop = detectLoop(index);
    if (loop) {
        loops_.push_back(*loop);
        graph_.optimize();
    }
    updateCorrection();
    return loop;
}

std::optional<LoopDetection> LoopClosure::detectLoop(int index) {
    const Keyframe& current = keyframes_[index];
    if (current.global_descriptor.size() == 0) return std::nullopt;

    // Candidates: older keyframes with the most similar global descriptor
    std::vector<std::pair<float, int>> scored;
    for (int k = 0; k + parameters_.recent_keyframes_excluded < index; ++k) {
        if (keyframes_[k].global_descriptor.size() == 0 || keyframes_[k].landmark_rows.size() < static_cast<std::size_t>(parameters_.min_inliers)) {
            ++statistics_.keyframes_without_landmarks;
            continue;
        }
        const float similarity = keyframes_[k].global_descriptor.dot(current.global_descriptor);
        statistics_.best_global_similarity = std::max(statistics_.best_global_similarity, static_cast<double>(similarity));
        if (similarity >= parameters_.min_global_similarity) scored.emplace_back(similarity, k);
        else ++statistics_.below_global_similarity;
    }
    std::sort(scored.begin(), scored.end(), std::greater<>());
    if (static_cast<int>(scored.size()) > parameters_.candidates) scored.resize(parameters_.candidates);

    std::optional<LoopDetection> best;
    Pose best_T_cand_curr;
    for (const auto& [similarity, k] : scored) {
        const Keyframe& candidate = keyframes_[k];

        // Mutual nearest neighbours between the landmarks of the candidate and all keypoints of the current keyframe
        Eigen::MatrixXf landmark_descriptors(candidate.landmark_rows.size(), candidate.descriptors.cols());
        for (std::size_t r = 0; r < candidate.landmark_rows.size(); ++r) landmark_descriptors.row(static_cast<Eigen::Index>(r)) = candidate.descriptors.row(candidate.landmark_rows[r]);
        const Eigen::MatrixXf similarity_matrix = landmark_descriptors * current.descriptors.transpose();

        std::vector<Vec3> points;
        std::vector<Vec2> observations;
        for (Eigen::Index r = 0; r < similarity_matrix.rows(); ++r) {
            Eigen::Index c;
            const float value = similarity_matrix.row(r).maxCoeff(&c);
            Eigen::Index r_back;
            similarity_matrix.col(c).maxCoeff(&r_back);
            if (r_back != r || value < parameters_.min_descriptor_similarity) continue;
            points.push_back(candidate.landmarks_camera[static_cast<std::size_t>(r)]);
            observations.push_back(current.observations[static_cast<std::size_t>(c)]);
        }
        if (static_cast<int>(points.size()) < parameters_.min_inliers) {
            ++statistics_.too_few_matches;
            continue;
        }

        // Pose of the candidate camera in the current camera, then of the current IMU in the candidate IMU
        const std::optional<PnPResult> pnp = solvePnPRansac(points, observations, parameters_.pnp_threshold / focal_, parameters_.min_inliers);
        if (!pnp) {
            ++statistics_.pnp_failed;
            continue;
        }
        const Pose T_cand_curr = T_I_C_ * pnp->T_C_W.inverse() * T_I_C_.inverse();

        // Reject loops implying an implausible drift: compare with the relative odometry pose
        const Pose T_cand_curr_vio = candidate.T_W_I_vio.inverse() * current.T_W_I_vio;
        if ((T_cand_curr.t - T_cand_curr_vio.t).norm() > parameters_.max_correction_distance) {
            ++statistics_.implausible_correction;
            continue;
        }

        const int inliers = static_cast<int>(pnp->inliers.size());
        if (!best || inliers > best->inliers) {
            best = LoopDetection{index, k, inliers};
            best_T_cand_curr = T_cand_curr;
        }
    }
    if (best) graph_.addEdge(best->candidate, best->keyframe, best_T_cand_curr, parameters_.loop_sigma_p, parameters_.loop_sigma_yaw, true);
    return best;
}

Pose LoopClosure::keyframeCorrection(std::size_t index) const {
    // Drift at a keyframe: optimized pose vs odometry pose, in yaw and translation
    const PoseGraph4Dof::Node& node = graph_.nodes()[index];
    const Pose& vio = keyframes_[index].T_W_I_vio;
    const Mat3 R = rotationZ(node.yaw - yawOf(vio.R));
    return Pose(R, node.p - R * vio.t);
}

void LoopClosure::updateCorrection() { correction_ = keyframeCorrection(keyframes_.size() - 1); }

Pose LoopClosure::correct(const Pose& T_W_I) const { return correction_ * T_W_I; }

Pose LoopClosure::correctAt(double timestamp, const Pose& T_W_I) const {
    if (keyframes_.empty()) return T_W_I;
    // First keyframe after the timestamp
    const auto after = std::upper_bound(keyframes_.begin(), keyframes_.end(), timestamp,
                                        [](double t, const Keyframe& keyframe) { return t < keyframe.timestamp; });
    if (after == keyframes_.end()) return correct(T_W_I);
    const std::size_t next = static_cast<std::size_t>(after - keyframes_.begin());
    if (next == 0) return keyframeCorrection(0) * T_W_I;
    const Pose a = keyframeCorrection(next - 1), b = keyframeCorrection(next);
    const double s = (timestamp - keyframes_[next - 1].timestamp) / (keyframes_[next].timestamp - keyframes_[next - 1].timestamp);
    double yaw_a = yawOf(a.R), yaw_b = yawOf(b.R);
    while (yaw_b - yaw_a > EIGEN_PI) yaw_b -= 2.0 * EIGEN_PI;
    while (yaw_b - yaw_a < -EIGEN_PI) yaw_b += 2.0 * EIGEN_PI;
    const Pose correction(rotationZ((1.0 - s) * yaw_a + s * yaw_b), (1.0 - s) * a.t + s * b.t);
    return correction * T_W_I;
}

}  // namespace msckf
