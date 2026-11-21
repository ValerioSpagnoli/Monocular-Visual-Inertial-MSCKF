#include "msckf/msckf.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>

namespace msckf {

namespace {

constexpr int kImuDim = 15;
constexpr int kCloneDim = 6;

/// Removes `count` rows and columns starting at `start` from a square matrix.
MatX removeBlock(const MatX& P, int start, int count) {
    const int n = static_cast<int>(P.rows());
    const int tail = n - start - count;
    MatX out(n - count, n - count);
    out.topLeftCorner(start, start) = P.topLeftCorner(start, start);
    out.topRightCorner(start, tail) = P.topRightCorner(start, tail);
    out.bottomLeftCorner(tail, start) = P.bottomLeftCorner(tail, start);
    out.bottomRightCorner(tail, tail) = P.bottomRightCorner(tail, tail);
    return out;
}

/// First index of the maximum (as torch.argmax).
template <typename Vector>
Eigen::Index argmax(const Vector& v) {
    Eigen::Index best = 0;
    for (Eigen::Index i = 1; i < v.size(); ++i)
        if (v(i) > v(best)) best = i;
    return best;
}

}  // namespace

Msckf::Msckf(const Parameters& parameters) : parameters_(parameters) {
    parameters_.min_number_of_frames_to_be_lost = std::max(parameters.min_number_of_frames_to_be_lost, 1);
    parameters_.min_number_of_frames_to_be_tracked = std::max(parameters.min_number_of_frames_to_be_tracked, 2);

    invK_ = parameters_.K.inverse();
    // Residuals are in normalized image coordinates (K^-1 x): convert the pixel noise accordingly
    sigma_normalized_ = parameters_.sigma_image / (0.5 * (parameters_.K(0, 0) + parameters_.K(1, 1)));
    max_triangulation_error_ = parameters_.max_triangulation_error / (0.5 * (parameters_.K(0, 0) + parameters_.K(1, 1)));

    imu_.gravity = parameters_.gravity;
    covariance_ = parameters_.initial_covariance_diagonal.asDiagonal();

    Qc_.block<3, 3>(0, 0) = Mat3::Identity() * std::pow(parameters_.gyroscope_noise_density, 2);
    Qc_.block<3, 3>(3, 3) = Mat3::Identity() * std::pow(parameters_.gyroscope_random_walk, 2);
    Qc_.block<3, 3>(6, 6) = Mat3::Identity() * std::pow(parameters_.accelerometer_noise_density, 2);
    Qc_.block<3, 3>(9, 9) = Mat3::Identity() * std::pow(parameters_.accelerometer_random_walk, 2);
}

//* IMU
//* ========================================================================================== *//

void Msckf::addImu(const ImuMeasurement& measurement) {
    if (!first_image_arrived_) {
        imu_buffer_.push_back(measurement);
        return;
    }
    if (!imu_.initialized) {
        imu_.initialize(imu_buffer_, parameters_.estimate_initial_gyroscope_bias);
        if (parameters_.initial_tilt_covariance_from_noise && imu_buffer_.size() > 1) {
            // The gravity direction comes from the mean of N noisy samples: sigma_f = density / sqrt(dt) per sample,
            // so the tilt error has variance sigma_f^2 / (N g^2)
            const double n = static_cast<double>(imu_buffer_.size());
            const double dt = (imu_buffer_.back().timestamp - imu_buffer_.front().timestamp) / (n - 1.0);
            if (dt > 0.0) {
                const double variance = std::pow(parameters_.accelerometer_noise_density, 2) / dt / (n * imu_.gravity.squaredNorm());
                covariance_.topLeftCorner<3, 3>() += variance * Mat3::Identity();
            }
        }
        for (const auto& buffered : imu_buffer_) processImu(buffered);
    }
    processImu(measurement);
}

void Msckf::processImu(const ImuMeasurement& measurement) {
    const double dt = measurement.timestamp - imu_.timestamp;
    imu_.timestamp = measurement.timestamp;
    imu_.id += 1;

    const Vec3 angular_velocity = measurement.angular_velocity - imu_.gyroscope_bias;
    const Vec3 linear_acceleration = measurement.linear_acceleration - imu_.accelerometer_bias;
    imu_.integrate(linear_acceleration, angular_velocity, dt);

    // Continuous-time error dynamics, state ordering [δθ, δbg, δv, δba, δp]
    const Mat3& R = imu_.T_W_I.R;
    MatX F = MatX::Zero(kImuDim, kImuDim);
    F.block<3, 3>(0, 0) = -skew(angular_velocity);
    F.block<3, 3>(0, 3) = -Mat3::Identity();
    F.block<3, 3>(6, 0) = -R * skew(linear_acceleration);
    F.block<3, 3>(6, 9) = -R;
    F.block<3, 3>(12, 6) = Mat3::Identity();

    MatX G = MatX::Zero(kImuDim, 12);
    G.block<3, 3>(0, 0) = -Mat3::Identity();
    G.block<3, 3>(3, 3) = Mat3::Identity();
    G.block<3, 3>(6, 6) = -R;
    G.block<3, 3>(9, 9) = Mat3::Identity();

    // Transition matrix (third-order Taylor expansion), then observability constraint
    const MatX Fdt = F * dt;
    const MatX Fdt2 = Fdt * Fdt;
    const MatX Fdt3 = Fdt2 * Fdt;
    MatX Phi = MatX::Identity(kImuDim, kImuDim) + Fdt + 0.5 * Fdt2 + (1.0 / 6.0) * Fdt3;
    Phi = observabilityConstrainedTransition(Phi, dt, imu_.T_W_I_null, imu_.v_W_I_null, imu_.T_W_I, imu_.v_W_I, imu_.gravity);

    const MatX Q = Phi * G * Qc_ * G.transpose() * Phi.transpose() * dt;
    const int n = static_cast<int>(covariance_.rows());
    covariance_.topLeftCorner(kImuDim, kImuDim) = Phi * covariance_.topLeftCorner(kImuDim, kImuDim) * Phi.transpose() + Q;
    if (n > kImuDim) {
        covariance_.topRightCorner(kImuDim, n - kImuDim) = Phi * covariance_.topRightCorner(kImuDim, n - kImuDim);
        covariance_.bottomLeftCorner(n - kImuDim, kImuDim) = covariance_.topRightCorner(kImuDim, n - kImuDim).transpose();
    }
    covariance_ = 0.5 * (covariance_ + covariance_.transpose()).eval();

    imu_.T_W_I_null = imu_.T_W_I;
    imu_.v_W_I_null = imu_.v_W_I;
}

//* Images
//* ========================================================================================== *//

void Msckf::addImage(const TrackedFrame& frame) {
    updated_feature_ids_.clear();
    gated_feature_ids_.clear();
    first_image_arrived_ = true;
    if (!imu_.initialized || parameters_.only_imu) return;

    augmentState();
    addCameraMeasurements(frame);
    if (last_disparity_ >= 0.0 && last_disparity_ < parameters_.zupt_max_disparity) zeroVelocityUpdate();
    processFeatures();
    if (static_cast<int>(cameras_.size()) > parameters_.max_number_of_camera_states) pruneCameraStates();
}

void Msckf::augmentState() {
    const Pose& T_I_C = parameters_.T_I_C;
    CameraClone clone;
    clone.T_W_C = imu_.T_W_I * T_I_C;
    clone.T_W_C_null = clone.T_W_C;
    cameras_[imu_.id] = clone;

    const int n = static_cast<int>(covariance_.rows());
    MatX J = MatX::Zero(kCloneDim, n);
    J.block<3, 3>(0, 0) = T_I_C.R.transpose();
    J.block<3, 3>(3, 0) = -imu_.T_W_I.R * skew(T_I_C.t);  // p_W_C = p_W_I + R_W_I exp(δθ) t_I_C
    J.block<3, 3>(3, 12) = Mat3::Identity();

    MatX P(n + kCloneDim, n + kCloneDim);
    P.topLeftCorner(n, n) = covariance_;
    P.topRightCorner(n, kCloneDim) = covariance_ * J.transpose();
    P.bottomLeftCorner(kCloneDim, n) = J * covariance_;
    P.bottomRightCorner(kCloneDim, kCloneDim) = J * covariance_ * J.transpose();
    covariance_ = 0.5 * (P + P.transpose());
}

int Msckf::cameraIndex(long long camera_id) const {
    int index = 0;
    for (const auto& [id, clone] : cameras_) {
        if (id == camera_id) return index;
        ++index;
    }
    throw std::out_of_range("unknown camera clone");
}

void Msckf::addCameraMeasurements(const TrackedFrame& frame) {
    const long long current_camera_id = imu_.id;
    const CameraClone& camera = cameras_.at(current_camera_id);

    // Observations of known features extend their tracks, the others start new features
    std::vector<double> disparities;
    std::set<long long> observed;
    for (const FeatureObservation& observation : frame.observations) {
        observed.insert(observation.id);
        const Vec3 W_v = camera.T_W_C.R * (invK_ * Vec3(observation.keypoint.x(), observation.keypoint.y(), 1.0));
        Feature& feature = features_[observation.id];
        if (!feature.keypoints.empty()) disparities.push_back((observation.keypoint - feature.keypoints.back()).norm());
        feature.keypoints.push_back(observation.keypoint);
        feature.descriptors.push_back(observation.descriptor);
        feature.scores.push_back(observation.score);
        feature.camera_ids.push_back(current_camera_id);
        feature.lines.push_back({camera.T_W_C.t, W_v, observation.score});
        feature.tracked_for_n_frames += 1;
        feature.lost_for_n_frames = 0;
    }
    for (auto& [id, feature] : features_)
        if (!observed.count(id)) feature.lost_for_n_frames += 1;

    // Median displacement (as np.median: mean of the two central values for an even count)
    last_disparity_ = -1.0;
    if (!disparities.empty()) {
        std::sort(disparities.begin(), disparities.end());
        const std::size_t half = disparities.size() / 2;
        last_disparity_ = disparities.size() % 2 ? disparities[half] : 0.5 * (disparities[half - 1] + disparities[half]);
    }
}

void Msckf::zeroVelocityUpdate() {
    // Static platform: v_W_I = 0 is a measurement. Features without parallax cannot be triangulated, so without it
    // the filter would drift with the IMU alone while static.
    const Eigen::Index n = covariance_.rows();
    MatX H = MatX::Zero(3, n);
    H.block<3, 3>(0, 6) = Mat3::Identity();
    const MatX R = parameters_.zupt_sigma_velocity * parameters_.zupt_sigma_velocity * MatX::Identity(3, 3);
    const VecX r = -imu_.v_W_I;
    const MatX S = H * covariance_ * H.transpose() + R;
    // Small image motion alone is not enough (e.g. moving towards distant points): the zero-velocity hypothesis must
    // also pass a chi-square test against the current velocity estimate
    if (r.dot(S.ldlt().solve(r)) > chi2Quantile(3)) return;
    const MatX K = S.ldlt().solve(H * covariance_).transpose();
    correct(K, H, R, K * r);
    ++zero_velocity_updates_;
}

//* Update
//* ========================================================================================== *//

void Msckf::processFeatures() {
    std::vector<long long> candidates;
    for (const auto& [id, feature] : features_) candidates.push_back(id);
    std::vector<long long> valid, lost;
    getValidFeatures(candidates, valid, lost);
    if (!valid.empty()) {
        update(valid);
        removeFeatures(lost);
    }
}

void Msckf::getValidFeatures(const std::vector<long long>& candidates, std::vector<long long>& valid, std::vector<long long>& lost,
                             bool use_all_tracked) {
    for (long long id : candidates) {
        Feature& feature = features_.at(id);
        const bool is_lost = feature.lost_for_n_frames >= parameters_.min_number_of_frames_to_be_lost;
        if (is_lost && feature.tracked_for_n_frames < parameters_.min_number_of_frames_to_be_tracked) {
            lost.push_back(id);
            continue;
        }

        bool enough_parallax = false;
        if (parameters_.use_parallax && feature.lines.size() > 1) {
            const double parallax = angleBetweenDirections(feature.lines.front().direction, feature.lines.back().direction) * 180.0 / EIGEN_PI;
            enough_parallax = parallax > parameters_.min_parallax;
        }
        // Still-tracked features are also used when their track is long enough, or when their observations are about
        // to be lost with the pruned clones, so that their information is not wasted
        const int observations = static_cast<int>(feature.camera_ids.size());
        const bool long_track = parameters_.max_track_length > 0 && observations >= parameters_.max_track_length;
        const bool forced = use_all_tracked && observations >= parameters_.min_number_of_frames_to_be_tracked;
        if (!is_lost && !enough_parallax && !long_track && !forced) continue;

        // The midpoint of the stored rays only initializes the Gauss-Newton triangulation, which uses the current
        // estimates of the camera clones. Features that cannot be triangulated reliably are not used.
        std::vector<Pose> poses;
        for (long long camera_id : feature.camera_ids) poses.push_back(cameras_.at(camera_id).T_W_C);
        const TriangulationResult triangulation = triangulateFeature(poses, feature.keypoints, invK_, intersectionOfLines(feature.lines));
        if (!triangulation.point || triangulation.rms > max_triangulation_error_) {
            ++triangulation_rejections_;
            if (is_lost) lost.push_back(id);
            continue;
        }

        feature.point = *triangulation.point;
        triangulated_points_[id] = feature.point;
        valid.push_back(id);
        if (is_lost) lost.push_back(id);
    }
}

bool Msckf::residualAndJacobian(const Feature& feature, VecX& r_o, MatX& H_o) const {
    const int m = static_cast<int>(feature.camera_ids.size());
    const int n = static_cast<int>(covariance_.rows());
    VecX r(2 * m);
    MatX H_x = MatX::Zero(2 * m, n);
    MatX H_f(2 * m, 3);
    const Vec3& W_f = feature.point;

    for (int i = 0; i < m; ++i) {
        const CameraClone& camera = cameras_.at(feature.camera_ids[i]);
        const Mat3 R_C_W = camera.T_W_C.R.transpose();
        const Vec3 C_f = R_C_W * (W_f - camera.T_W_C.t);

        const Vec3 z = invK_ * Vec3(feature.keypoints[i].x(), feature.keypoints[i].y(), 1.0);
        r.segment<2>(2 * i) = z.head<2>() / z.z() - C_f.head<2>() / C_f.z();

        Eigen::Matrix<double, 2, 3> J;
        J << 1.0 / C_f.z(), 0.0, -C_f.x() / (C_f.z() * C_f.z()),
             0.0, 1.0 / C_f.z(), -C_f.y() / (C_f.z() * C_f.z());
        Eigen::Matrix<double, 2, 6> H_c;
        H_c.leftCols<3>() = J * skew(C_f);
        H_c.rightCols<3>() = -J * R_C_W;

        // Observability constraint: remove the component along the unobservable rotation about gravity,
        // evaluated at the first estimate of the clone (Hesch et al. 2014)
        Eigen::Matrix<double, 6, 1> u;
        u.head<3>() = camera.T_W_C_null.R.transpose() * imu_.gravity;
        u.tail<3>() = skew(W_f - camera.T_W_C_null.t) * imu_.gravity;
        const double den = u.dot(u);
        if (den > 1e-6) H_c -= (H_c * u) * u.transpose() / den;

        // The feature Jacobian must come from the modified block (H_f = -H_p)
        H_f.block<2, 3>(2 * i, 0) = -H_c.rightCols<3>();
        H_x.block<2, 6>(2 * i, kImuDim + kCloneDim * cameraIndex(feature.camera_ids[i])) = H_c;
    }

    // Left null space of H_f: last 2m-3 columns of a complete QR
    const Eigen::HouseholderQR<MatX> qr(H_f);
    const MatX Q = qr.householderQ();
    const MatX A = Q.rightCols(2 * m - 3);
    r_o = A.transpose() * r;
    H_o = A.transpose() * H_x;
    return true;
}

bool Msckf::gatingTest(const VecX& r, const MatX& H) const {
    const MatX S = H * covariance_ * H.transpose() + sigma_normalized_ * sigma_normalized_ * MatX::Identity(H.rows(), H.rows());
    const double gamma = r.dot(S.ldlt().solve(r));
    return gamma <= chi2Quantile(static_cast<int>(r.size()));
}

void Msckf::update(const std::vector<long long>& feature_ids) {
    std::vector<VecX> residuals;
    std::vector<MatX> jacobians;
    for (long long id : feature_ids) {
        VecX r_o;
        MatX H_o;
        residualAndJacobian(features_.at(id), r_o, H_o);
        if (!gatingTest(r_o, H_o)) {
            ++gating_rejections_;
            gated_feature_ids_.push_back(id);
            continue;
        }
        updated_feature_ids_.push_back(id);
        residuals.push_back(std::move(r_o));
        jacobians.push_back(std::move(H_o));
    }
    if (residuals.empty()) return;

    Eigen::Index rows = 0;
    for (const auto& r : residuals) rows += r.size();
    const Eigen::Index n = covariance_.rows();
    MatX H_X(rows, n);
    VecX r_o(rows);
    Eigen::Index row = 0;
    for (std::size_t k = 0; k < residuals.size(); ++k) {
        H_X.middleRows(row, residuals[k].size()) = jacobians[k];
        r_o.segment(row, residuals[k].size()) = residuals[k];
        row += residuals[k].size();
    }

    // Measurement compression (QR) when there are more rows than states
    const double sigma2 = sigma_normalized_ * sigma_normalized_;
    MatX T_H;
    VecX r_n;
    MatX R_n;
    if (rows > n) {
        const Eigen::HouseholderQR<MatX> qr(H_X);
        const MatX Q = qr.householderQ() * MatX::Identity(rows, n);
        T_H = qr.matrixQR().topRows(n).triangularView<Eigen::Upper>();
        r_n = Q.transpose() * r_o;
        R_n = sigma2 * (Q.transpose() * Q);
    } else {
        T_H = H_X;
        r_n = r_o;
        R_n = sigma2 * MatX::Identity(rows, rows);
    }

    const MatX S = T_H * covariance_ * T_H.transpose() + R_n;
    const MatX K = S.ldlt().solve(T_H * covariance_).transpose();  // P T_H^T S^-1 (P and S symmetric)
    const VecX delta_x = K * r_n;
    correct(K, T_H, R_n, delta_x);

    std::vector<long long> used;
    for (long long id : feature_ids)
        if (std::find(updated_feature_ids_.begin(), updated_feature_ids_.end(), id) != updated_feature_ids_.end()) used.push_back(id);
    discardUsedObservations(used);
}

void Msckf::correct(const MatX& K, const MatX& T_H, const MatX& R_n, const VecX& delta_x) {
    const Eigen::Index n = covariance_.rows();
    const MatX I_KH = MatX::Identity(n, n) - K * T_H;
    const MatX P = I_KH * covariance_ * I_KH.transpose() + K * R_n * K.transpose();  // Joseph form
    covariance_ = 0.5 * (P + P.transpose());

    // Local orientation errors: R_true = R_est exp(δθ). New poses: the null states keep the first estimates
    imu_.T_W_I = Pose(orthonormalize(imu_.T_W_I.R * expSO3(delta_x.segment<3>(0))), imu_.T_W_I.t + delta_x.segment<3>(12));
    imu_.gyroscope_bias += delta_x.segment<3>(3);
    imu_.v_W_I += delta_x.segment<3>(6);
    imu_.accelerometer_bias += delta_x.segment<3>(9);

    int index = 0;
    for (auto& [id, clone] : cameras_) {
        const VecX delta = delta_x.segment(kImuDim + kCloneDim * index++, kCloneDim);
        clone.T_W_C = Pose(orthonormalize(clone.T_W_C.R * expSO3(delta.head<3>())), clone.T_W_C.t + delta.tail<3>());
    }
}

void Msckf::discardUsedObservations(const std::vector<long long>& feature_ids) {
    // Each measurement enters the EKF only once: features still tracked keep only their last observation
    for (long long id : feature_ids) {
        const auto it = features_.find(id);
        if (it == features_.end() || it->second.camera_ids.size() <= 1) continue;
        Feature& feature = it->second;
        feature.keypoints.erase(feature.keypoints.begin(), feature.keypoints.end() - 1);
        feature.descriptors.erase(feature.descriptors.begin(), feature.descriptors.end() - 1);
        feature.scores.erase(feature.scores.begin(), feature.scores.end() - 1);
        feature.camera_ids.erase(feature.camera_ids.begin(), feature.camera_ids.end() - 1);
        feature.lines.erase(feature.lines.begin(), feature.lines.end() - 1);
        feature.tracked_for_n_frames = 1;
    }
}

//* State management
//* ========================================================================================== *//

void Msckf::pruneCameraStates() {
    // Number of feature observations per camera, in order of first appearance, then the two poorest cameras
    std::vector<std::pair<long long, int>> counts;
    for (const auto& [id, feature] : features_) {
        for (long long camera_id : feature.camera_ids) {
            auto it = std::find_if(counts.begin(), counts.end(), [&](const auto& c) { return c.first == camera_id; });
            if (it == counts.end()) counts.emplace_back(camera_id, 1);
            else ++it->second;
        }
    }
    std::stable_sort(counts.begin(), counts.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
    std::vector<long long> to_remove;
    for (std::size_t i = 0; i < counts.size() && i < 2; ++i) to_remove.push_back(counts[i].first);

    std::vector<long long> candidates;
    for (const auto& [id, feature] : features_)
        for (long long camera_id : feature.camera_ids)
            if (std::find(to_remove.begin(), to_remove.end(), camera_id) != to_remove.end()) {
                candidates.push_back(id);
                break;
            }

    std::vector<long long> valid, lost;
    getValidFeatures(candidates, valid, lost, parameters_.update_features_on_prune);
    if (!valid.empty()) update(valid);
    removeCameras(to_remove);
}

KeyframeObservations Msckf::currentObservations(double min_parallax_degrees) const {
    KeyframeObservations observations;
    std::vector<Eigen::VectorXf> descriptors;
    for (const auto& [id, feature] : features_) {
        if (feature.camera_ids.empty() || feature.camera_ids.back() != imu_.id) continue;
        observations.keypoints.push_back(feature.keypoints.back());
        descriptors.push_back(feature.descriptors.back());
        std::optional<Vec3> point;
        const auto triangulated = triangulated_points_.find(id);
        if (triangulated != triangulated_points_.end()) {
            point = triangulated->second;
        } else if (feature.camera_ids.size() >= 3 &&
                   angleBetweenDirections(feature.lines.front().direction, feature.lines.back().direction) * 180.0 / EIGEN_PI >= min_parallax_degrees) {
            std::vector<Pose> poses;
            for (long long camera_id : feature.camera_ids) poses.push_back(cameras_.at(camera_id).T_W_C);
            const TriangulationResult result = triangulateFeature(poses, feature.keypoints, invK_, intersectionOfLines(feature.lines));
            if (result.point && result.rms <= max_triangulation_error_) point = result.point;
        }
        observations.points.push_back(point);
    }
    if (!descriptors.empty()) {
        observations.descriptors.resize(static_cast<Eigen::Index>(descriptors.size()), descriptors.front().size());
        for (std::size_t k = 0; k < descriptors.size(); ++k) observations.descriptors.row(static_cast<Eigen::Index>(k)) = descriptors[k].transpose();
    }
    return observations;
}

void Msckf::removeFeatures(const std::vector<long long>& feature_ids) {
    for (long long id : feature_ids) {
        features_.erase(id);
        triangulated_points_.erase(id);  // bounded memory on long trajectories
    }
    std::vector<long long> unused_cameras;
    for (const auto& [camera_id, clone] : cameras_) {
        bool observed = false;
        for (const auto& [id, feature] : features_)
            if (std::find(feature.camera_ids.begin(), feature.camera_ids.end(), camera_id) != feature.camera_ids.end()) {
                observed = true;
                break;
            }
        if (!observed) unused_cameras.push_back(camera_id);
    }
    removeCameras(unused_cameras);
}

void Msckf::removeCameras(const std::vector<long long>& camera_ids) {
    for (long long camera_id : camera_ids) {
        covariance_ = removeBlock(covariance_, kImuDim + kCloneDim * cameraIndex(camera_id), kCloneDim);
        cameras_.erase(camera_id);
    }

    std::vector<long long> empty_features;
    for (auto& [id, feature] : features_) {
        for (long long camera_id : camera_ids) {
            const auto it = std::find(feature.camera_ids.begin(), feature.camera_ids.end(), camera_id);
            if (it == feature.camera_ids.end()) continue;
            const auto k = it - feature.camera_ids.begin();
            feature.keypoints.erase(feature.keypoints.begin() + k);
            feature.descriptors.erase(feature.descriptors.begin() + k);
            feature.scores.erase(feature.scores.begin() + k);
            feature.camera_ids.erase(it);
            feature.lines.erase(feature.lines.begin() + k);
        }
        if (feature.camera_ids.empty()) empty_features.push_back(id);
    }
    for (long long id : empty_features) {
        features_.erase(id);
        triangulated_points_.erase(id);
    }
}

}  // namespace msckf
