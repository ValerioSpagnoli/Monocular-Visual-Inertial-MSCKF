// XFeat front-end against a reference saved once from the PyTorch implementation.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <fstream>

#include <opencv2/imgcodecs.hpp>

#include "msckf_frontend/xfeat.hpp"

using namespace msckf;

TEST_CASE("XFeat front-end matches the PyTorch implementation") {
    // Reference saved from XFeat.detectAndCompute (PyTorch, CPU) on an undistorted EuRoC image
    const std::string data = MSCKF_TEST_DATA_DIR;
    const cv::Mat image = cv::imread(data + "/euroc_v1_01_frame_1000.png", cv::IMREAD_GRAYSCALE);
    REQUIRE(!image.empty());
    std::ifstream file(data + "/euroc_v1_01_frame_1000_xfeat.bin", std::ios::binary);
    int32_t n = 0, d = 0;
    file.read(reinterpret_cast<char*>(&n), 4);
    file.read(reinterpret_cast<char*>(&d), 4);
    std::vector<float> keypoints(2 * n), scores(n), descriptors(static_cast<std::size_t>(n) * d);
    file.read(reinterpret_cast<char*>(keypoints.data()), keypoints.size() * 4);
    file.read(reinterpret_cast<char*>(scores.data()), scores.size() * 4);
    file.read(reinterpret_cast<char*>(descriptors.data()), descriptors.size() * 4);

    XFeatExtractor::Options options;
    options.top_k = n;
    const FrameFeatures features = XFeatExtractor(MSCKF_MODEL_PATH, options).extract(image);
    REQUIRE(static_cast<int>(features.keypoints.size()) == n);
    double max_keypoint = 0.0, max_score = 0.0, max_descriptor = 0.0;
    for (int k = 0; k < n; ++k) {
        max_keypoint = std::max(max_keypoint, (features.keypoints[k] - Vec2(keypoints[2 * k], keypoints[2 * k + 1])).cwiseAbs().maxCoeff());
        max_score = std::max(max_score, static_cast<double>(std::abs(features.scores[k] - scores[k])));
        for (int c = 0; c < d; ++c) max_descriptor = std::max(max_descriptor, static_cast<double>(std::abs(features.descriptors(k, c) - descriptors[k * d + c])));
    }
    CHECK(max_keypoint < 1e-3);
    CHECK(max_score < 1e-4);
    CHECK(max_descriptor < 1e-4);
}
