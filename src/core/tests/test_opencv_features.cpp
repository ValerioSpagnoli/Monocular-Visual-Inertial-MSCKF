// Classical OpenCV features in the format of the filter.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <opencv2/imgcodecs.hpp>

#include "msckf/opencv_features.hpp"

using namespace msckf;

TEST_CASE("OpenCV features give spread keypoints and normalized descriptors") {
    const cv::Mat image = cv::imread(std::string(MSCKF_TEST_DATA_DIR) + "/euroc_v1_01_frame_1000.png", cv::IMREAD_GRAYSCALE);
    REQUIRE(!image.empty());
    for (const auto type : {OpenCvExtractor::Type::Orb, OpenCvExtractor::Type::Sift, OpenCvExtractor::Type::Akaze}) {
        OpenCvExtractor::Options options;
        options.type = type;
        const FrameFeatures features = OpenCvExtractor(options).extract(image);
        INFO("type " << static_cast<int>(type));
        CHECK(features.keypoints.size() > 200);
        CHECK(static_cast<int>(features.keypoints.size()) <= options.top_k);
        REQUIRE(features.descriptors.rows() == static_cast<Eigen::Index>(features.keypoints.size()));
        int top_half = 0, left_half = 0;
        for (Eigen::Index r = 0; r < features.descriptors.rows(); ++r) {
            CHECK(features.descriptors.row(r).norm() == Catch::Approx(1.0f).margin(1e-5));
            const Vec2& u = features.keypoints[static_cast<std::size_t>(r)];
            CHECK((u.x() >= 0 && u.y() >= 0 && u.x() < image.cols && u.y() < image.rows));
            top_half += u.y() < image.rows / 2;
            left_half += u.x() < image.cols / 2;
        }
        // The grid spreads the keypoints over the image
        const double n = static_cast<double>(features.keypoints.size());
        CHECK(top_half / n > 0.25);
        CHECK(top_half / n < 0.75);
        CHECK(left_half / n > 0.25);
        CHECK(left_half / n < 0.75);
    }
}

TEST_CASE("Binary descriptors: cosine similarity equals 1 - 2 Hamming / bits") {
    const cv::Mat image = cv::imread(std::string(MSCKF_TEST_DATA_DIR) + "/euroc_v1_01_frame_1000.png", cv::IMREAD_GRAYSCALE);
    REQUIRE(!image.empty());
    const FrameFeatures features = OpenCvExtractor(OpenCvExtractor::Options{}).extract(image);  // ORB
    REQUIRE(features.descriptors.cols() == 256);
    for (Eigen::Index a = 0; a < 20; ++a) {
        const Eigen::RowVectorXf x = features.descriptors.row(a), y = features.descriptors.row(a + 1);
        int hamming = 0;
        for (Eigen::Index b = 0; b < 256; ++b) hamming += (x(b) > 0) != (y(b) > 0);
        CHECK(x.dot(y) == Catch::Approx(1.0 - 2.0 * hamming / 256.0).margin(1e-5));
    }
}
