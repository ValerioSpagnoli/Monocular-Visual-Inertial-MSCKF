#include <catch2/catch_test_macros.hpp>

#include <random>

#include "msckf/loop_closure.hpp"
#include "msckf/pnp.hpp"
#include "msckf/pose_graph.hpp"

using namespace msckf;

namespace {

Vec3 randomVector(std::mt19937& rng, double scale) {
    std::normal_distribution<double> normal(0.0, scale);
    return {normal(rng), normal(rng), normal(rng)};
}

/// Pose of a camera on a circle of the given radius, looking at the center (IMU = camera in these tests).
Pose circlePose(double angle, double radius) {
    const Vec3 center(radius * std::cos(angle), radius * std::sin(angle), 1.0);
    const Vec3 z = (Vec3(0.0, 0.0, 1.0) - center).normalized();  // optical axis towards the scene
    const Vec3 x = z.cross(Vec3(0.0, 0.0, 1.0)).normalized();
    const Vec3 y = z.cross(x);
    Mat3 R;
    R << x, y, z;
    return {R, center};
}

}  // namespace

TEST_CASE("PnP with outliers") {
    std::mt19937 rng(3);
    const Pose T_C_W(expSO3(randomVector(rng, 0.5)), randomVector(rng, 1.0));
    std::uniform_real_distribution<double> uniform(-2.0, 2.0);
    std::vector<Vec3> points;
    std::vector<Vec2> observations;
    int outliers = 0;
    for (int k = 0; k < 60; ++k) {
        const Vec3 p_C(uniform(rng), uniform(rng), 4.0 + uniform(rng));
        points.push_back(T_C_W.inverse().transform(p_C));
        Vec2 x = p_C.head<2>() / p_C.z() + randomVector(rng, 0.5 / 400.0).head<2>();
        if (k % 3 == 0) {  // one third of wrong matches
            x += Vec2(0.1, -0.08);
            ++outliers;
        }
        observations.push_back(x);
    }
    const auto result = solvePnPRansac(points, observations, 3.0 / 400.0, 20);
    REQUIRE(result.has_value());
    CHECK(static_cast<int>(result->inliers.size()) >= 60 - outliers - 2);
    CHECK(static_cast<int>(result->inliers.size()) <= 60 - outliers);
    CHECK((result->T_C_W.t - T_C_W.t).norm() < 0.02);
    CHECK(logSO3(result->T_C_W.R.transpose() * T_C_W.R).norm() < 0.005);
}

TEST_CASE("4-dof pose graph closes a drifting loop") {
    // Odometry around a circle with yaw and scale drift; a loop edge says the last node is back at the first one
    const int n = 40;
    std::vector<Pose> truth, odometry;
    for (int k = 0; k <= n; ++k) truth.push_back(circlePose(2.0 * EIGEN_PI * k / n, 3.0));
    Pose drifted = truth[0];
    odometry.push_back(drifted);
    for (int k = 1; k <= n; ++k) {
        const Pose step = truth[k - 1].inverse() * truth[k];
        drifted = drifted * Pose(step.R, step.t * 1.01);
        drifted = Pose(rotationZ(0.004) * drifted.R, drifted.t);
        odometry.push_back(drifted);
    }

    PoseGraph4Dof graph;
    for (int k = 0; k <= n; ++k) graph.addNode(odometry[k]);
    for (int k = 1; k <= n; ++k) graph.addEdge(k - 1, k, odometry[k - 1].inverse() * odometry[k], 0.02, 0.005, false);
    graph.addEdge(0, n, truth[0].inverse() * truth[n], 0.02, 0.005, true);

    const double error_before = (odometry[n].t - truth[n].t).norm();
    graph.optimize();
    const double error_after = (graph.nodes()[n].p - truth[n].t).norm();
    CHECK(error_before > 0.3);
    CHECK(error_after < 0.05);
}

TEST_CASE("Loop closure recognizes a revisited place and corrects the drift") {
    std::mt19937 rng(5);
    const Mat3 K = (Mat3() << 400.0, 0.0, 320.0, 0.0, 400.0, 240.0, 0.0, 0.0, 1.0).finished();

    // Landmarks around the center of the circle, each with its own random unit descriptor
    std::uniform_real_distribution<double> uniform(-1.5, 1.5);
    std::vector<Vec3> landmarks;
    Eigen::MatrixXf descriptors(600, 64);
    for (int k = 0; k < 600; ++k) {
        landmarks.emplace_back(uniform(rng), uniform(rng), 1.0 + uniform(rng));
        descriptors.row(k) = Eigen::VectorXf::Random(64).normalized().transpose();
    }

    LoopClosureParameters parameters;
    parameters.recent_keyframes_excluded = 10;
    parameters.min_global_similarity = 0.0;
    LoopClosure loop_closure(parameters, K, Pose());

    // A bit more than one lap; the odometry drifts in yaw and translation
    const int n = 50;
    bool loop_found = false;
    double error_odometry = 0.0, error_corrected = 0.0;
    for (int k = 0; k <= n + 3; ++k) {
        const Pose truth = circlePose(2.0 * EIGEN_PI * k / n, 3.0);
        const Pose drift(rotationZ(0.003 * k), Vec3(0.01 * k, -0.005 * k, 0.0));
        const Pose odometry = drift * truth;

        KeyframeObservations observations;
        std::vector<int> visible;
        for (int l = 0; l < static_cast<int>(landmarks.size()); ++l) {
            const Vec3 p_C = truth.inverse().transform(landmarks[l]);
            if (p_C.z() < 0.5) continue;
            const Vec2 u = (K * (p_C / p_C.z())).head<2>();
            if (u.x() < 0 || u.x() > 640 || u.y() < 0 || u.y() > 480) continue;
            observations.keypoints.push_back(u);
            observations.points.push_back(drift.transform(landmarks[l]));  // triangulated in the drifting VIO frame
            visible.push_back(l);
        }
        observations.descriptors.resize(static_cast<Eigen::Index>(visible.size()), 64);
        for (std::size_t r = 0; r < visible.size(); ++r) observations.descriptors.row(static_cast<Eigen::Index>(r)) = descriptors.row(visible[r]);

        if (loop_closure.addKeyframe(0.1 * k, odometry, observations)) loop_found = true;
        if (k == n + 3) {
            error_odometry = (odometry.t - truth.t).norm();
            error_corrected = (loop_closure.correct(odometry).t - truth.t).norm();
        }
    }
    REQUIRE(loop_found);
    CHECK(error_odometry > 0.3);
    CHECK(error_corrected < 0.1 * error_odometry);

    // A later keyframe without any loop must keep the drift correction (not fall back to the odometry)
    const Pose truth = circlePose(2.0 * EIGEN_PI * (n + 4) / n, 3.0);
    const Pose odometry = Pose(rotationZ(0.003 * (n + 4)), Vec3(0.01 * (n + 4), -0.005 * (n + 4), 0.0)) * truth;
    CHECK_FALSE(loop_closure.addKeyframe(0.1 * (n + 4), odometry, KeyframeObservations{}).has_value());
    CHECK((loop_closure.correct(odometry).t - truth.t).norm() < 0.1 * (odometry.t - truth.t).norm());
}
