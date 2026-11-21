// Tests of the desktop application components: XFeat front-end against a PyTorch reference, PCHIP, IMU generator.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>

#include <opencv2/imgcodecs.hpp>

#include "../app/dataset.hpp"
#include "msckf_frontend/xfeat.hpp"

using namespace msckf;
namespace fs = std::filesystem;

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

TEST_CASE("PCHIP interpolates without overshoot") {
    const std::vector<double> t{0.0, 1.0, 2.0, 3.0, 4.0};
    const std::vector<Vec3> v{Vec3(0, 0, 1), Vec3(0, 1, 1), Vec3(0, 1, 2), Vec3(1, 3, 2), Vec3(2, 3, 5)};
    const Pchip pchip(t, v);
    for (std::size_t k = 0; k < t.size(); ++k) CHECK((pchip.value(t[k]) - v[k]).norm() < 1e-12);
    for (double s = 0.0; s <= 4.0; s += 0.01) {
        const Vec3 x = pchip.value(s);
        CHECK(x.x() >= -1e-12); CHECK(x.x() <= 2.0 + 1e-12);  // monotone data stays within the data range
        CHECK(x.y() >= -1e-12); CHECK(x.y() <= 3.0 + 1e-12);
        if (s > 0.01 && s < 3.99) CHECK((pchip.derivative(s) - (pchip.value(s + 1e-6) - pchip.value(s - 1e-6)) / 2e-6).norm() < 1e-5);
    }
    CHECK(pchip.value(1.5).y() == Catch::Approx(1.0));  // flat segment stays flat
}

namespace {

/// Temporary TartanAir-like sequence with a smooth moving trajectory.
std::string makeSequence(const std::string& name) {
    const fs::path root = fs::temp_directory_path() / ("msckf_app_tests_" + name);
    fs::remove_all(root);
    fs::create_directories(root / "tartanair" / "M000" / "cam");
    std::ofstream(root / "tartanair" / "camera_info.csv") << "R00,R01,R02,R10,R11,R12,R20,R21,R22,fx,fy,px,py,w,h\n0,0,1,1,0,0,0,1,0,320,320,320,240,640,480\n";
    std::ofstream trajectory(root / "tartanair" / "M000" / "trajectory.csv");
    trajectory << "x y z qx qy qz qw\n";
    for (int k = 0; k < 120; ++k) {
        char image[32];
        std::snprintf(image, sizeof(image), "%06d.png", k);
        std::ofstream(root / "tartanair" / "M000" / "cam" / image);
        const double s = 0.05 * k;
        const Eigen::Quaterniond q(Eigen::AngleAxisd(0.8 * std::sin(s), Vec3::UnitZ()) * Eigen::AngleAxisd(0.3 * std::sin(1.3 * s), Vec3::UnitY()) *
                                   Eigen::AngleAxisd(0.2 * std::cos(0.7 * s), Vec3::UnitX()));
        trajectory << 2 * std::cos(s) << ' ' << 1.5 * std::sin(s) << ' ' << 0.3 * std::sin(2 * s) << ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
    }
    return root.string();
}

}  // namespace

TEST_CASE("Noise-free generated IMU reproduces the ground truth") {
    const Sequence sequence = loadPhotorealistic(makeSequence("exact"), "tartanair", "M000", ImuNoise{});
    ImuState imu;
    imu.T_W_I = *sequence.ground_truth.front();
    imu.v_W_I = Vec3::Zero();  // static start
    for (std::size_t k = 1; k < sequence.imu.size(); ++k)
        imu.integrate(sequence.imu[k].linear_acceleration, sequence.imu[k].angular_velocity, sequence.imu[k].timestamp - sequence.imu[k - 1].timestamp);
    const Pose& truth = *sequence.ground_truth.back();
    CHECK((imu.T_W_I.t - truth.t).norm() < 1e-6);
    CHECK(logSO3(imu.T_W_I.R.transpose() * truth.R).norm() < 1e-9);
}

TEST_CASE("Generated IMU noise has the declared statistics") {
    const ImuNoise noise{0.01, 0.001, 0.0, 0.0};  // white noise only
    const std::string root = makeSequence("noise");
    const Sequence clean = loadPhotorealistic(root, "tartanair", "M000", ImuNoise{});
    const Sequence noisy = loadPhotorealistic(root, "tartanair", "M000", noise);
    const double dt = clean.imu[2].timestamp - clean.imu[1].timestamp;
    double accel = 0.0, gyro = 0.0;
    const std::size_t n = clean.imu.size() - 1;
    for (std::size_t k = 1; k <= n; ++k) {
        accel += (noisy.imu[k].linear_acceleration - clean.imu[k].linear_acceleration).squaredNorm();
        gyro += (noisy.imu[k].angular_velocity - clean.imu[k].angular_velocity).squaredNorm();
    }
    CHECK(std::sqrt(accel / (3 * n)) == Catch::Approx(noise.accelerometer_noise_density / std::sqrt(dt)).epsilon(0.05));
    CHECK(std::sqrt(gyro / (3 * n)) == Catch::Approx(noise.gyroscope_noise_density / std::sqrt(dt)).epsilon(0.05));
}
