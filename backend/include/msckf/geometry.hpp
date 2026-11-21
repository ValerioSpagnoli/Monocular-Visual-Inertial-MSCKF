#pragma once

#include <optional>
#include <vector>

#include "msckf/math.hpp"

namespace msckf {

/// Ray in the world frame (camera center, viewing direction, weight).
struct Line {
    Vec3 base = Vec3::Zero();
    Vec3 direction = Vec3::Zero();
    double confidence = 1.0;
};

/// Weighted least-squares point closest to all lines (midpoint triangulation).
Vec3 intersectionOfLines(const std::vector<Line>& lines);

/// Angle between two directions [rad].
double angleBetweenDirections(const Vec3& d1, const Vec3& d2);

/// F = K^-T [t]x R K^-1 for T_C1_C2 (pose of camera 2 in camera 1): x1^T F x2 = 0 for pixels of the same point.
Mat3 fundamentalMatrix(const Pose& T_C1_C2, const Mat3& invK);

/// Sampson distance [px] of the match (keypoint_1, keypoint_2) from the constraint x1^T F x2 = 0.
double sampsonDistance(const Mat3& F, const Vec2& keypoint_1, const Vec2& keypoint_2);

struct TriangulationResult {
    std::optional<Vec3> point;  // empty if the point is not in front of every camera
    double rms = 0.0;           // RMS reprojection error in normalized coordinates
};

/// Gauss-Newton (Levenberg-Marquardt) triangulation in inverse depth anchored to the first camera
/// (Mourikis and Roumeliotis 2007): x = [X/Z, Y/Z, 1/Z], minimizing the normalized reprojection error.
TriangulationResult triangulateFeature(const std::vector<Pose>& poses, const std::vector<Vec2>& keypoints, const Mat3& invK,
                                       const Vec3& initial_point, int max_iterations = 10);

/// OC-EKF transition matrix (Hesch et al. 2014) from the unconstrained one, evaluated with the previous (null)
/// state and the propagated state. The paper uses the JPL rotation C = R_W_I^T, hence the transposes.
MatX observabilityConstrainedTransition(const MatX& Phi, double dt, const Pose& T_W_I_null, const Vec3& v_W_I_null,
                                        const Pose& T_W_I, const Vec3& v_W_I, const Vec3& gravity);

/// Quantile of the chi-square distribution with `dof` degrees of freedom (default 95 %).
double chi2Quantile(int dof, double probability = 0.95);

}  // namespace msckf
