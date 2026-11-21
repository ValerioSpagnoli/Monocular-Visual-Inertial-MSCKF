#pragma once

#include <vector>

#include "msckf/math.hpp"

namespace msckf {

/// Yaw (rotation about the world z axis, aligned with gravity) of R = Rz(yaw) Ry(pitch) Rx(roll).
double yawOf(const Mat3& R);
Mat3 rotationZ(double yaw);

/// 4-dof pose graph (position and yaw, as in VINS-Mono): roll and pitch are observable from gravity, so a VIO
/// drifts only in these 4 directions. Each node keeps its roll and pitch fixed to the odometry value.
class PoseGraph4Dof {
public:
    struct Node {
        Vec3 p = Vec3::Zero();
        double yaw = 0.0;
        Mat3 R_roll_pitch = Mat3::Identity();  // R = Rz(yaw) * R_roll_pitch
        Pose pose() const { return {rotationZ(yaw) * R_roll_pitch, p}; }
    };

    struct Edge {
        int i = 0, j = 0;
        Vec3 delta_p = Vec3::Zero();  // p_j - p_i in the yaw frame of node i: Rz(yaw_i)^T (p_j - p_i)
        double delta_yaw = 0.0;
        double sigma_p = 0.05;        // [m]
        double sigma_yaw = 0.02;      // [rad]
        bool robust = false;          // Huber loss (loop closures, which may be wrong)
    };

    /// Adds a node initialized with the odometry pose; returns its index.
    int addNode(const Pose& T_W_I);

    /// Edge from the relative pose T_i_j (node j in node i) measured in the body frames.
    void addEdge(int i, int j, const Pose& T_i_j, double sigma_p, double sigma_yaw, bool robust);

    /// Gauss-Newton with the first node fixed (gauge); Huber weights on robust edges.
    void optimize(int iterations = 10, double huber_threshold = 1.0);

    const std::vector<Node>& nodes() const { return nodes_; }
    const std::vector<Edge>& edges() const { return edges_; }

private:
    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
};

}  // namespace msckf
