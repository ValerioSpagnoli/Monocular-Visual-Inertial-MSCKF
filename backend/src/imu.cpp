#include "msckf/imu.hpp"

#include <Eigen/Geometry>

namespace msckf {

void ImuState::integrate(const Vec3& linear_acceleration, const Vec3& angular_velocity, double dt) {
    const Mat3 R_W_I = T_W_I.R;
    const Vec3 p_W_I = T_W_I.t;

    Mat3 dR = Mat3::Identity();
    const double angle = angular_velocity.norm() * dt;
    if (angle > 0.0) {
        const Mat3 K = skew(angular_velocity.normalized());
        dR = Mat3::Identity() + std::sin(angle) * K + (1.0 - std::cos(angle)) * (K * K);
    }

    const Vec3 a_W = R_W_I * linear_acceleration + gravity;
    T_W_I = Pose(R_W_I * dR, p_W_I + v_W_I * dt + 0.5 * a_W * dt * dt);
    v_W_I = v_W_I + a_W * dt;
}

void ImuState::initialize(const std::vector<ImuMeasurement>& measurements, bool estimate_gyroscope_bias) {
    initialized = true;

    Vec3 mean_angular_velocity = Vec3::Zero();
    Vec3 mean_linear_acceleration = Vec3::Zero();
    for (const auto& m : measurements) {
        mean_angular_velocity += m.angular_velocity;
        mean_linear_acceleration += m.linear_acceleration;
    }
    mean_angular_velocity /= static_cast<double>(measurements.size());
    mean_linear_acceleration /= static_cast<double>(measurements.size());

    // At rest the accelerometer measures f = R_W_I^T (0 - g): the specific force points up
    const Vec3 W_up = -gravity.normalized();
    const Vec3 I_up = mean_linear_acceleration.normalized();
    const Mat3 R_W_I = Eigen::Quaterniond::FromTwoVectors(I_up, W_up).toRotationMatrix();

    T_W_I = Pose(R_W_I, Vec3::Zero());
    if (estimate_gyroscope_bias) gyroscope_bias = mean_angular_velocity;
}

}  // namespace msckf
