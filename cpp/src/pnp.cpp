#include "msckf/pnp.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>

namespace msckf {

namespace {

/// Direct linear transform from >= 6 correspondences: P = [R | t] up to scale, projected onto SO(3).
std::optional<Pose> dlt(const std::vector<Vec3>& points, const std::vector<Vec2>& observations, const std::vector<int>& indices) {
    const int n = static_cast<int>(indices.size());
    MatX A = MatX::Zero(2 * n, 12);
    for (int k = 0; k < n; ++k) {
        const Vec3& X = points[indices[k]];
        const Vec2& x = observations[indices[k]];
        const Eigen::Vector4d Xh(X.x(), X.y(), X.z(), 1.0);
        A.block<1, 4>(2 * k, 0) = Xh.transpose();
        A.block<1, 4>(2 * k, 8) = -x.x() * Xh.transpose();
        A.block<1, 4>(2 * k + 1, 4) = Xh.transpose();
        A.block<1, 4>(2 * k + 1, 8) = -x.y() * Xh.transpose();
    }
    const Eigen::JacobiSVD<MatX> svd(A, Eigen::ComputeFullV);
    const VecX p = svd.matrixV().col(11);
    Eigen::Matrix<double, 3, 4> P;
    P << p.segment<4>(0).transpose(), p.segment<4>(4).transpose(), p.segment<4>(8).transpose();

    // Scale and sign: the left 3x3 block must be a rotation, points in front of the camera
    const Eigen::JacobiSVD<Mat3> svd_R(P.leftCols<3>(), Eigen::ComputeFullU | Eigen::ComputeFullV);
    const double scale = svd_R.singularValues().mean();
    if (scale < 1e-12) return std::nullopt;
    Mat3 R = svd_R.matrixU() * svd_R.matrixV().transpose();
    Vec3 t = P.col(3) / scale;
    if (R.determinant() < 0) {
        R = -R;
        t = -t;
    }
    // A wrong overall sign puts the points behind the camera
    int in_front = 0;
    for (int k : indices) in_front += (R * points[k] + t).z() > 0 ? 1 : -1;
    if (in_front < 0) return std::nullopt;
    return Pose(R, t);
}

double reprojectionError(const Pose& T, const Vec3& X, const Vec2& x) {
    const Vec3 Xc = T.transform(X);
    if (Xc.z() <= 1e-6) return std::numeric_limits<double>::infinity();
    return (Xc.head<2>() / Xc.z() - x).norm();
}

}  // namespace

Pose refinePose(const Pose& T_C_W, const std::vector<Vec3>& points, const std::vector<Vec2>& observations, int iterations) {
    // Perturbation on the left: T <- [exp(δθ) | δt] * T
    Pose T = T_C_W;
    const int n = static_cast<int>(points.size());
    for (int iteration = 0; iteration < iterations; ++iteration) {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
        for (int k = 0; k < n; ++k) {
            const Vec3 Xc = T.transform(points[k]);
            if (Xc.z() <= 1e-6) continue;
            const Vec2 r = observations[k] - Xc.head<2>() / Xc.z();
            Eigen::Matrix<double, 2, 3> J_proj;
            J_proj << 1.0 / Xc.z(), 0.0, -Xc.x() / (Xc.z() * Xc.z()),
                      0.0, 1.0 / Xc.z(), -Xc.y() / (Xc.z() * Xc.z());
            Eigen::Matrix<double, 3, 6> J_pose;
            J_pose.leftCols<3>() = -skew(Xc);
            J_pose.rightCols<3>() = Mat3::Identity();
            const Eigen::Matrix<double, 2, 6> J = J_proj * J_pose;
            H += J.transpose() * J;
            b += J.transpose() * r;
        }
        const Eigen::Matrix<double, 6, 1> delta = H.ldlt().solve(b);
        if (!delta.allFinite()) break;
        T = Pose(orthonormalize(expSO3(delta.head<3>()) * T.R), expSO3(delta.head<3>()) * T.t + delta.tail<3>());
        if (delta.norm() < 1e-10) break;
    }
    return T;
}

std::optional<PnPResult> solvePnPRansac(const std::vector<Vec3>& points, const std::vector<Vec2>& observations, double threshold,
                                        int min_inliers, int iterations, unsigned seed) {
    const int n = static_cast<int>(points.size());
    if (n < 6 || n < min_inliers) return std::nullopt;

    std::mt19937 rng(seed);
    std::vector<int> all(n);
    std::iota(all.begin(), all.end(), 0);

    std::vector<int> best_inliers;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::vector<int> sample = all;
        std::shuffle(sample.begin(), sample.end(), rng);
        sample.resize(6);
        const std::optional<Pose> T = dlt(points, observations, sample);
        if (!T) continue;
        std::vector<int> inliers;
        for (int k = 0; k < n; ++k)
            if (reprojectionError(*T, points[k], observations[k]) < threshold) inliers.push_back(k);
        if (inliers.size() > best_inliers.size()) best_inliers = std::move(inliers);
    }
    if (static_cast<int>(best_inliers.size()) < min_inliers) return std::nullopt;

    // Refine on the inliers (DLT on all of them, then Gauss-Newton), and recompute the inlier set once
    PnPResult result;
    const std::optional<Pose> initial = dlt(points, observations, best_inliers);
    if (!initial) return std::nullopt;
    std::vector<Vec3> inlier_points;
    std::vector<Vec2> inlier_observations;
    for (int k : best_inliers) {
        inlier_points.push_back(points[k]);
        inlier_observations.push_back(observations[k]);
    }
    result.T_C_W = refinePose(*initial, inlier_points, inlier_observations);
    for (int k = 0; k < n; ++k)
        if (reprojectionError(result.T_C_W, points[k], observations[k]) < threshold) result.inliers.push_back(k);
    if (static_cast<int>(result.inliers.size()) < min_inliers) return std::nullopt;
    return result;
}

}  // namespace msckf
