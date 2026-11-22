#pragma once

#include <vector>

#include "msckf/math.hpp"

namespace msckf {

struct ImuMeasurement {
    double timestamp = 0.0;
    Vec3 angular_velocity = Vec3::Zero();     // [rad/s]
    Vec3 linear_acceleration = Vec3::Zero();  // specific force f = R_W_I^T (a - g) [m/s^2]
};

/// IMU state: pose and velocity in the world frame, biases, and the first-estimate (null) values used by
/// the observability constraint.
struct ImuState {
    long long id = 0;  // number of processed IMU samples, also used as camera-clone id
    double timestamp = 0.0;
    Pose T_W_I;        // current IMU pose in the world frame
    Vec3 v_W_I = Vec3::Zero();
    Vec3 gyroscope_bias = Vec3::Zero();
    Vec3 accelerometer_bias = Vec3::Zero();
    Vec3 gravity{0.0, 0.0, -9.81};

    Pose T_W_I_null;
    Vec3 v_W_I_null = Vec3::Zero();

    bool initialized = false;

    /// Strapdown integration over dt with bias-corrected measurements (same scheme as IMU.integrate):
    /// R <- R exp(w dt), a_W = R f + g, p <- p + v dt + a_W dt^2 / 2, v <- v + a_W dt.
    void integrate(const Vec3& linear_acceleration, const Vec3& angular_velocity, double dt);

    /// Initial attitude from the mean specific force (static start): smallest rotation aligning the measured
    /// up direction with the world up direction. Optionally the gyroscope bias is the mean angular velocity.
    void initialize(const std::vector<ImuMeasurement>& measurements, bool estimate_gyroscope_bias);
};

}  // namespace msckf
