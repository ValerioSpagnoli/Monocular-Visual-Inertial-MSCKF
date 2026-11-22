#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "dataset.hpp"
#include "evaluation.hpp"
#include "msckf/vio_system.hpp"

namespace rerun { class RecordingStream; }

namespace msckf {

/// Streams the estimator to Rerun: 3D map (trajectories, body poses, camera frustum with the image, sliding window,
/// landmarks, 3-sigma ellipsoid, loop-corrected trajectory), camera image with features and tracks colored by status,
/// and markdown panels with metrics and filter status. Disabled when constructed without a destination.
class Visualizer {
public:
    /// destination: empty optional = disabled, "" = viewer only, path = viewer and .rrd file.
    Visualizer(const Sequence& sequence, const std::optional<std::string>& destination);
    ~Visualizer();

    /// Called at every IMU step; image is non-null after an image has been processed.
    void step(long long index, double timestamp, const VioSystem& system, const std::optional<Pose>& ground_truth, const cv::Mat* image,
              const Evaluator& evaluator);

private:
    void logFrame(const VioSystem& system, const cv::Mat& image, double timestamp, const Evaluator& evaluator);
    void logLoopClosure(const LoopClosure& loop_closure);

    std::unique_ptr<rerun::RecordingStream> rec_;
    std::vector<std::array<float, 3>> estimate_trajectory_, ground_truth_trajectory_, corrected_trajectory_;
    std::vector<std::pair<double, Pose>> odometry_;  // timestamped odometry poses, to recompute the corrected trajectory
    std::map<long long, Vec3> landmarks_;  // history of triangulated points (the filter keeps only the live ones)
    std::optional<Pose> last_ground_truth_;
    long long frames_ = 0;
    std::size_t logged_keyframes_ = 0, logged_loops_ = 0;
};

}  // namespace msckf
