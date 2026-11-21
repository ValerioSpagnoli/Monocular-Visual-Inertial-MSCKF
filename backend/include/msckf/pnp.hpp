#pragma once

#include <optional>
#include <vector>

#include "msckf/math.hpp"

namespace msckf {

struct PnPResult {
    Pose T_C_W;                    // pose of the reference frame of the points in the camera: p_C = R p_W + t
    std::vector<int> inliers;      // indices of the correspondences consistent with the pose
};

/// Pose of a calibrated camera from 3D points and their normalized image coordinates (z = 1 plane):
/// RANSAC over 6-point DLT solutions, then Gauss-Newton refinement of the reprojection error on the inliers.
/// `threshold` is the maximum reprojection error in normalized coordinates (pixels / focal length).
std::optional<PnPResult> solvePnPRansac(const std::vector<Vec3>& points, const std::vector<Vec2>& observations, double threshold,
                                        int min_inliers, int iterations = 200, unsigned seed = 0);

/// Gauss-Newton refinement of T_C_W minimizing the normalized reprojection error of the given correspondences.
Pose refinePose(const Pose& T_C_W, const std::vector<Vec3>& points, const std::vector<Vec2>& observations, int iterations = 10);

}  // namespace msckf
