#pragma once

#include <cmath>
#include <Eigen/Dense>

namespace msckf {

using Mat3 = Eigen::Matrix3d;
using Vec3 = Eigen::Vector3d;
using Vec2 = Eigen::Vector2d;
using MatX = Eigen::MatrixXd;
using VecX = Eigen::VectorXd;

/// Skew-symmetric matrix: skew(a) * b = a x b.
inline Mat3 skew(const Vec3& w) {
    Mat3 S;
    S << 0.0, -w.z(), w.y(),
         w.z(), 0.0, -w.x(),
        -w.y(), w.x(), 0.0;
    return S;
}

/// Exponential map of SO(3) (Rodrigues). Rotation vectors with norm <= 1e-8 give the identity,
/// as np.isclose(norm, 0) in the Python implementation.
inline Mat3 expSO3(const Vec3& phi) {
    const double angle = phi.norm();
    if (angle <= 1e-8) return Mat3::Identity();
    const Mat3 K = skew(phi);
    return Mat3::Identity() + (std::sin(angle) / angle) * K + ((1.0 - std::cos(angle)) / (angle * angle)) * (K * K);
}

/// Logarithm map of SO(3): rotation vector of R.
inline Vec3 logSO3(const Mat3& R) {
    const Eigen::AngleAxisd angle_axis(R);
    return angle_axis.angle() * angle_axis.axis();
}

/// Closest rotation matrix (U V^T of the SVD).
inline Mat3 orthonormalize(const Mat3& R) {
    const Eigen::JacobiSVD<Mat3> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * svd.matrixV().transpose();
}

/// Rigid transform: p_parent = R * p_child + t.
struct Pose {
    Mat3 R = Mat3::Identity();
    Vec3 t = Vec3::Zero();

    Pose() = default;
    Pose(const Mat3& R_, const Vec3& t_) : R(R_), t(t_) {}

    Pose inverse() const { return {R.transpose(), -R.transpose() * t}; }
    Pose operator*(const Pose& other) const { return {R * other.R, R * other.t + t}; }
    Vec3 transform(const Vec3& p) const { return R * p + t; }
};

}  // namespace msckf
