#pragma once

#include <optional>
#include <vector>

#include "msckf/math.hpp"
#include "msckf/pose_graph.hpp"

namespace msckf {

struct LoopClosureParameters {
    double keyframe_distance = 0.3;        // new keyframe after this translation [m] ...
    double keyframe_angle = 15.0;          // ... or this rotation [deg]
    int recent_keyframes_excluded = 20;    // loop candidates must be older than this many keyframes
    int candidates = 5;                    // keyframes checked by geometric verification (best global similarity)
    double min_global_similarity = 0.7;    // cosine similarity of the global descriptors
    double min_descriptor_similarity = 0.9;
    int min_inliers = 25;                  // PnP inliers to accept a loop
    double pnp_threshold = 3.0;            // reprojection error of a PnP inlier [px]
    double max_correction_distance = 5.0;  // reject loops implying a larger drift correction [m]
    double odometry_sigma_p = 0.02;        // [m] relative pose between consecutive keyframes
    double odometry_sigma_yaw = 0.005;     // [rad]
    double loop_sigma_p = 0.05;            // [m]
    double loop_sigma_yaw = 0.02;          // [rad]
};

/// Observations of the current image used by a keyframe: keypoints [px], descriptors and, when the filter has
/// triangulated the feature, its world point (VIO frame).
struct KeyframeObservations {
    std::vector<Vec2> keypoints;
    Eigen::MatrixXf descriptors;
    std::vector<std::optional<Vec3>> points;
};

struct LoopDetection {
    int keyframe = -1;   // new keyframe
    int candidate = -1;  // older keyframe recognized
    int inliers = 0;
};

/// Place recognition and 4-dof pose-graph optimization on top of the VIO (as in VINS-Mono). The filter is not
/// modified: the result is a drift correction applied to its output.
class LoopClosure {
public:
    LoopClosure(const LoopClosureParameters& parameters, const Mat3& K, const Pose& T_I_C);

    /// True if the IMU pose moved enough since the last keyframe.
    bool needsKeyframe(const Pose& T_W_I) const;

    /// Adds a keyframe with the VIO pose of the IMU; searches for a loop and optimizes the graph if one is found.
    std::optional<LoopDetection> addKeyframe(double timestamp, const Pose& T_W_I, const KeyframeObservations& observations);

    /// VIO pose corrected with the drift estimated at the last keyframe.
    Pose correct(const Pose& T_W_I) const;

    const PoseGraph4Dof& graph() const { return graph_; }
    const std::vector<LoopDetection>& loops() const { return loops_; }

private:
    struct Keyframe {
        double timestamp = 0.0;
        Pose T_W_I_vio;                       // odometry pose when the keyframe was created
        std::vector<Vec2> observations;       // normalized coordinates of all keypoints
        Eigen::MatrixXf descriptors;          // normalized rows
        Eigen::VectorXf global_descriptor;    // normalized mean of the descriptors
        std::vector<int> landmark_rows;       // rows of `descriptors` with a 3D point
        std::vector<Vec3> landmarks_camera;   // their points in the camera frame of the keyframe
    };

    std::optional<LoopDetection> detectLoop(int index);
    void updateCorrection();

    LoopClosureParameters parameters_;
    Mat3 invK_;
    double focal_;
    Pose T_I_C_;
    std::vector<Keyframe> keyframes_;
    PoseGraph4Dof graph_;
    std::vector<LoopDetection> loops_;
    Pose correction_;  // T_W_corrected <- correction_ * T_W_vio (yaw and translation only)
};

}  // namespace msckf
