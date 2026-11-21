#pragma once

#include <optional>

#include "msckf/loop_closure.hpp"
#include "msckf/msckf.hpp"

namespace msckf {

/// Entry point of the estimator: MSCKF odometry plus (optionally) loop closure on keyframes.
class VioSystem {
public:
    VioSystem(const Parameters& parameters, const LoopClosureParameters& loop_closure_parameters, bool enable_loop_closure);

    void addImu(const ImuMeasurement& measurement) { filter_.addImu(measurement); }

    /// Processes an image; returns the loop detected at this image, if any.
    std::optional<LoopDetection> addImage(double timestamp, const TrackedFrame& frame);

    /// Odometry pose of the IMU, and the same pose corrected with the loop closures.
    const Pose& odometryPose() const { return filter_.imu().T_W_I; }
    Pose correctedPose() const;

    const Msckf& filter() const { return filter_; }
    const LoopClosure* loopClosure() const { return loop_closure_ ? &*loop_closure_ : nullptr; }

private:
    Msckf filter_;
    std::optional<LoopClosure> loop_closure_;
};

}  // namespace msckf
