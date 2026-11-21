#include <catch2/catch_test_macros.hpp>

#include <map>
#include <random>

#include "msckf_frontend/tracker.hpp"

using namespace msckf;

namespace {

const Mat3 K = (Mat3() << 400.0, 0.0, 320.0, 0.0, 400.0, 240.0, 0.0, 0.0, 1.0).finished();

struct Scene {
    std::vector<Vec3> points;
    Eigen::MatrixXf descriptors;
};

Scene makeScene(std::mt19937& rng, int n) {
    std::uniform_real_distribution<double> uniform(-2.0, 2.0);
    Scene scene;
    scene.descriptors.resize(n, 64);
    for (int k = 0; k < n; ++k) {
        scene.points.emplace_back(uniform(rng), uniform(rng), 6.0 + uniform(rng));
        scene.descriptors.row(k) = Eigen::VectorXf::Random(64).normalized().transpose();
    }
    return scene;
}

/// Features of the scene seen from a camera at `center` looking along +z; `outliers` keypoints are moved far from
/// their true position while keeping their descriptor (wrong matches).
FrameFeatures observe(const Scene& scene, const Vec3& center, const std::vector<int>& outliers, std::vector<int>& indices) {
    FrameFeatures features;
    indices.clear();
    for (int k = 0; k < static_cast<int>(scene.points.size()); ++k) {
        const Vec3 p = scene.points[k] - center;
        Vec2 u = (K * (p / p.z())).head<2>();
        if (std::find(outliers.begin(), outliers.end(), k) != outliers.end()) u += Vec2(40.0, -30.0);
        features.keypoints.push_back(u);
        features.scores.push_back(1.0f);
        indices.push_back(k);
    }
    features.descriptors = scene.descriptors;
    return features;
}

}  // namespace

TEST_CASE("Tracker keeps the ids of the tracked points and rejects wrong matches") {
    std::mt19937 rng(7);
    const Scene scene = makeScene(rng, 80);
    Tracker tracker(Tracker::Options{});

    std::vector<int> indices;
    const Tracker::Result first = tracker.track(observe(scene, Vec3::Zero(), {}, indices));
    REQUIRE(first.frame.observations.size() == 80);
    CHECK(first.matched == 0);
    std::map<int, long long> id_of_point;
    for (std::size_t k = 0; k < indices.size(); ++k) id_of_point[indices[k]] = first.frame.observations[k].id;

    // The camera moves; 8 points are given a wrong position
    const std::vector<int> outliers{3, 11, 19, 27, 35, 43, 51, 59};
    const Tracker::Result second = tracker.track(observe(scene, Vec3(0.3, -0.1, 0.2), outliers, indices));
    CHECK(second.matched == 72);
    CHECK(second.rejected.size() == outliers.size());
    CHECK(tracker.rejectedMatches() == static_cast<long long>(outliers.size()));
    for (const FeatureObservation& observation : second.frame.observations) {
        // Every observation continues the track of its own point (rejected matches are dropped, not new tracks)
        const auto it = std::find_if(id_of_point.begin(), id_of_point.end(), [&](const auto& entry) { return entry.second == observation.id; });
        REQUIRE(it != id_of_point.end());
        CHECK(std::find(outliers.begin(), outliers.end(), it->first) == outliers.end());
    }
}

TEST_CASE("Tracker ends lost tracks and starts new ones") {
    std::mt19937 rng(8);
    const Scene a = makeScene(rng, 30), b = makeScene(rng, 30);
    Tracker::Options options;
    options.max_lost_frames = 2;
    Tracker tracker(options);
    std::vector<int> indices;

    tracker.track(observe(a, Vec3::Zero(), {}, indices));
    CHECK(tracker.activeTracks() == 30);
    // Two images of a different scene: the first tracks are lost twice and end, the new points start new tracks
    const Tracker::Result r1 = tracker.track(observe(b, Vec3::Zero(), {}, indices));
    CHECK(r1.matched == 0);
    CHECK(tracker.activeTracks() == 60);
    tracker.track(observe(b, Vec3(0.1, 0.0, 0.0), {}, indices));
    CHECK(tracker.activeTracks() == 30);
}
