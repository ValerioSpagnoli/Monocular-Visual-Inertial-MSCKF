#include "msckf/vio_system.hpp"

namespace msckf {

VioSystem::VioSystem(const Parameters& parameters, const LoopClosureParameters& loop_closure_parameters, bool enable_loop_closure)
    : filter_(parameters) {
    if (enable_loop_closure) loop_closure_.emplace(loop_closure_parameters, parameters.K, parameters.T_I_C);
}

std::optional<LoopDetection> VioSystem::addImage(double timestamp, const TrackedFrame& frame) {
    filter_.addImage(frame);
    if (!loop_closure_ || !filter_.imu().initialized || !loop_closure_->needsKeyframe(filter_.imu().T_W_I)) return std::nullopt;
    const KeyframeObservations observations = filter_.currentObservations();
    if (observations.keypoints.empty()) return std::nullopt;
    return loop_closure_->addKeyframe(timestamp, filter_.imu().T_W_I, observations);
}

Pose VioSystem::correctedPose() const {
    return loop_closure_ ? loop_closure_->correct(filter_.imu().T_W_I) : filter_.imu().T_W_I;
}

}  // namespace msckf
