#include "dataset.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

namespace msckf {

namespace {

/// Rows of numbers of a CSV/space-separated file; lines starting with '#' or not starting with a number are skipped.
std::vector<std::vector<std::string>> readTable(const std::string& path, char separator) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open " + path);
    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#' || !(std::isdigit(static_cast<unsigned char>(line[0])) || line[0] == '-' || line[0] == '.')) continue;
        std::vector<std::string> fields;
        std::stringstream stream(line);
        std::string field;
        while (std::getline(stream, field, separator))
            if (!field.empty()) fields.push_back(field);
        rows.push_back(std::move(fields));
    }
    return rows;
}

Pose matrixToPose(const std::vector<double>& data) {
    Eigen::Matrix4d T;
    for (int k = 0; k < 16; ++k) T(k / 4, k % 4) = data[static_cast<std::size_t>(k)];
    return {T.topLeftCorner<3, 3>(), T.topRightCorner<3, 1>()};
}

std::vector<double> yamlVector(const YAML::Node& node) { return node.as<std::vector<double>>(); }

Mat3 quaternionToRotation(double qw, double qx, double qy, double qz) {
    return Eigen::Quaterniond(qw, qx, qy, qz).normalized().toRotationMatrix();
}

}  // namespace

cv::Mat Sequence::loadImage(std::size_t index) const {
    cv::Mat image = cv::imread(images[index].path, cv::IMREAD_COLOR);
    if (image.empty()) throw std::runtime_error("cannot read " + images[index].path);
    if (!undistort_map_x.empty()) cv::remap(image, image, undistort_map_x, undistort_map_y, cv::INTER_LINEAR);
    return image;
}

//* EuRoC
//* ========================================================================================== *//

Sequence loadEuroc(const std::string& root, const std::string& name) {
    const std::string mav = root + "/euroc/" + name + "/mav0/";
    const YAML::Node camera = YAML::LoadFile(mav + "cam0/sensor.yaml");
    const YAML::Node imu = YAML::LoadFile(mav + "imu0/sensor.yaml");
    Sequence sequence;
    sequence.real_imu = true;
    sequence.align_ground_truth = true;

    //* Calibration: images are undistorted once so that the filter keeps its pinhole model
    const std::vector<double> intrinsics = yamlVector(camera["intrinsics"]);
    const std::vector<double> distortion = yamlVector(camera["distortion_coefficients"]);
    const std::vector<double> resolution = yamlVector(camera["resolution"]);
    sequence.width = static_cast<int>(resolution[0]);
    sequence.height = static_cast<int>(resolution[1]);
    double k_data[9] = {intrinsics[0], 0, intrinsics[2], 0, intrinsics[1], intrinsics[3], 0, 0, 1};
    const cv::Mat K_distorted = cv::Mat(3, 3, CV_64F, k_data).clone();
    const cv::Mat coefficients(distortion, true);
    const cv::Size size(sequence.width, sequence.height);
    const cv::Mat K = cv::getOptimalNewCameraMatrix(K_distorted, coefficients, size, 0.0);
    cv::initUndistortRectifyMap(K_distorted, coefficients, cv::Mat(), K, size, CV_32FC1, sequence.undistort_map_x, sequence.undistort_map_y);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) sequence.K(r, c) = K.at<double>(r, c);

    const Pose T_B_C = matrixToPose(yamlVector(camera["T_BS"]["data"]));
    const Pose T_B_I = matrixToPose(yamlVector(imu["T_BS"]["data"]));
    sequence.T_I_C = T_B_I.inverse() * T_B_C;
    sequence.imu_noise = {imu["accelerometer_noise_density"].as<double>(), imu["gyroscope_noise_density"].as<double>(),
                          imu["accelerometer_random_walk"].as<double>(), imu["gyroscope_random_walk"].as<double>()};

    //* IMU (timestamps in seconds from the first sample)
    const auto imu_rows = readTable(mav + "imu0/data.csv", ',');
    const long long t0 = std::stoll(imu_rows.front()[0]);
    for (const auto& row : imu_rows) {
        ImuMeasurement m;
        m.timestamp = (std::stoll(row[0]) - t0) * 1e-9;
        m.angular_velocity = Vec3(std::stod(row[1]), std::stod(row[2]), std::stod(row[3]));
        m.linear_acceleration = Vec3(std::stod(row[4]), std::stod(row[5]), std::stod(row[6]));
        sequence.imu.push_back(m);
    }

    //* Images
    for (const auto& row : readTable(mav + "cam0/data.csv", ',')) {
        const long long t = std::stoll(row[0]);
        if (t >= t0) sequence.images.push_back({(t - t0) * 1e-9, mav + "cam0/data/" + row[1]});
    }

    //* Ground truth resampled at the IMU timestamps (linear positions, slerp rotations)
    std::vector<double> gt_times;
    std::vector<Vec3> gt_positions;
    std::vector<Eigen::Quaterniond> gt_rotations;
    for (const auto& row : readTable(mav + "state_groundtruth_estimate0/data.csv", ',')) {
        gt_times.push_back((std::stoll(row[0]) - t0) * 1e-9);
        gt_positions.emplace_back(std::stod(row[1]), std::stod(row[2]), std::stod(row[3]));
        gt_rotations.emplace_back(Eigen::Quaterniond(std::stod(row[4]), std::stod(row[5]), std::stod(row[6]), std::stod(row[7])).normalized());
    }
    for (const auto& m : sequence.imu) {
        const auto upper = std::upper_bound(gt_times.begin(), gt_times.end(), m.timestamp);
        if (m.timestamp < gt_times.front() || m.timestamp > gt_times.back()) {
            sequence.ground_truth.emplace_back(std::nullopt);
            continue;
        }
        const std::size_t k = upper == gt_times.end() ? gt_times.size() - 1 : static_cast<std::size_t>(upper - gt_times.begin());
        const std::size_t j = k == 0 ? 0 : k - 1;
        const double span = gt_times[k] - gt_times[j];
        const double s = span > 0 ? (m.timestamp - gt_times[j]) / span : 0.0;
        sequence.ground_truth.emplace_back(Pose(gt_rotations[j].slerp(s, gt_rotations[k]).toRotationMatrix(), (1 - s) * gt_positions[j] + s * gt_positions[k]));
    }
    return sequence;
}

//* Photorealistic datasets with generated IMU
//* ========================================================================================== *//

Pchip::Pchip(const std::vector<double>& times, const std::vector<Vec3>& values) : times_(times), values_(values), slopes_(values.size()) {
    // Derivatives as scipy.interpolate.PchipInterpolator (Fritsch-Carlson, one-sided shape-preserving ends)
    const std::size_t n = times.size();
    if (n < 2) throw std::invalid_argument("PCHIP needs at least two points");
    std::vector<double> h(n - 1);
    std::vector<Vec3> m(n - 1);
    for (std::size_t k = 0; k + 1 < n; ++k) {
        h[k] = times[k + 1] - times[k];
        m[k] = (values[k + 1] - values[k]) / h[k];
    }
    auto sign = [](double x) { return (x > 0) - (x < 0); };
    for (int c = 0; c < 3; ++c) {
        if (n == 2) {
            slopes_[0](c) = slopes_[1](c) = m[0](c);
            continue;
        }
        for (std::size_t k = 1; k + 1 < n; ++k) {
            const double m0 = m[k - 1](c), m1 = m[k](c);
            if (sign(m0) != sign(m1) || m0 == 0.0 || m1 == 0.0) {
                slopes_[k](c) = 0.0;
            } else {
                const double w1 = 2 * h[k] + h[k - 1], w2 = h[k] + 2 * h[k - 1];
                slopes_[k](c) = (w1 + w2) / (w1 / m0 + w2 / m1);
            }
        }
        auto edge = [&](double h0, double h1, double m0, double m1) {
            double d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
            if (sign(d) != sign(m0)) d = 0.0;
            else if (sign(m0) != sign(m1) && std::abs(d) > 3 * std::abs(m0)) d = 3 * m0;
            return d;
        };
        slopes_[0](c) = edge(h[0], h[1], m[0](c), m[1](c));
        slopes_[n - 1](c) = edge(h[n - 2], h[n - 3], m[n - 2](c), m[n - 3](c));
    }
}

std::size_t Pchip::segment(double t) const {
    const auto upper = std::upper_bound(times_.begin(), times_.end(), t);
    if (upper == times_.begin()) return 0;
    return std::min(static_cast<std::size_t>(upper - times_.begin()) - 1, times_.size() - 2);
}

Vec3 Pchip::value(double t) const {
    const std::size_t k = segment(t);
    const double h = times_[k + 1] - times_[k], s = t - times_[k];
    const Vec3 m = (values_[k + 1] - values_[k]) / h;
    const Vec3 c2 = (3 * m - 2 * slopes_[k] - slopes_[k + 1]) / h;
    const Vec3 c3 = (slopes_[k] + slopes_[k + 1] - 2 * m) / (h * h);
    return values_[k] + slopes_[k] * s + c2 * s * s + c3 * s * s * s;
}

Vec3 Pchip::derivative(double t) const {
    const std::size_t k = segment(t);
    const double h = times_[k + 1] - times_[k], s = t - times_[k];
    const Vec3 m = (values_[k + 1] - values_[k]) / h;
    const Vec3 c2 = (3 * m - 2 * slopes_[k] - slopes_[k + 1]) / h;
    const Vec3 c3 = (slopes_[k] + slopes_[k + 1] - 2 * m) / (h * h);
    return slopes_[k] + 2 * c2 * s + 3 * c3 * s * s;
}

Sequence loadPhotorealistic(const std::string& root, const std::string& source, const std::string& name, const ImuNoise& noise, unsigned seed) {
    const std::string folder = root + "/" + source + "/" + name + "/";
    const bool tartanair = source == "tartanair";
    Sequence sequence;
    sequence.imu_noise = noise;

    //* Camera: intrinsics and rotation of the optical frame in the IMU frame
    const auto info = readTable(root + "/" + source + "/camera_info.csv", ',').front();
    std::vector<double> values;
    for (const auto& v : info) values.push_back(std::stod(v));
    Mat3 R_I_C;
    for (int k = 0; k < 9; ++k) R_I_C(k / 3, k % 3) = values[static_cast<std::size_t>(k)];
    sequence.T_I_C = Pose(R_I_C, Vec3::Zero());
    sequence.K << values[9], 0.0, values[11], 0.0, values[10], values[12], 0.0, 0.0, 1.0;
    sequence.width = static_cast<int>(values[13]);
    sequence.height = static_cast<int>(values[14]);

    //* Trajectory: TartanAir poses are in the NED camera frame (= IMU frame), PeringLab poses are camera poses
    std::vector<Pose> poses;
    for (const auto& row : readTable(folder + "trajectory.csv", tartanair ? ' ' : ',')) {
        if (tartanair) poses.emplace_back(quaternionToRotation(std::stod(row[6]), std::stod(row[3]), std::stod(row[4]), std::stod(row[5])),
                                          Vec3(std::stod(row[0]), std::stod(row[1]), std::stod(row[2])));
        else poses.emplace_back(quaternionToRotation(std::stod(row[4]), std::stod(row[5]), std::stod(row[6]), std::stod(row[7])),
                                Vec3(std::stod(row[1]), std::stod(row[2]), std::stod(row[3])));
        if (!tartanair) poses.back() = poses.back() * sequence.T_I_C.inverse();  // camera pose -> IMU pose
    }
    std::vector<std::string> image_paths;
    for (const auto& entry : fs::directory_iterator(folder + "cam")) image_paths.push_back(entry.path().string());
    std::sort(image_paths.begin(), image_paths.end());
    const std::size_t n = std::min(poses.size(), image_paths.size());

    //* Keyframes every 0.05 s in the frame of the first pose, after a static start of 0.1 s (gravity initialization)
    const Pose T_0_W = poses.front().inverse();
    constexpr double frame_period = 0.05, static_duration = 0.1;
    std::vector<double> key_times;
    std::vector<Vec3> key_positions;
    std::vector<Eigen::Quaterniond> key_rotations;
    for (int k = 0; k < 3; ++k) {
        key_times.push_back(k * frame_period);
        key_positions.push_back(Vec3::Zero());
        key_rotations.push_back(Eigen::Quaterniond::Identity());
        sequence.images.push_back({k * frame_period, image_paths[0]});
    }
    for (std::size_t i = 1; i < n; ++i) {
        const Pose T = T_0_W * poses[i];
        key_times.push_back(static_duration + i * frame_period);
        key_positions.push_back(T.t);
        key_rotations.emplace_back(T.R);
        sequence.images.push_back({static_duration + i * frame_period, image_paths[i]});
    }

    //* IMU at 9 samples per frame: rotations by slerp, positions with spline velocities and the trapezoidal rule, so
    //* that ImuState::integrate reproduces the ground truth exactly
    const Pchip spline(key_times, key_positions);
    constexpr int samples_per_frame = 9;
    std::vector<double> times;
    for (std::size_t k = 0; k + 1 < key_times.size(); ++k)
        for (int s = 0; s < samples_per_frame; ++s) times.push_back(key_times[k] + (key_times[k + 1] - key_times[k]) * s / samples_per_frame);
    times.push_back(key_times.back());

    std::vector<Mat3> rotations(times.size());
    std::vector<Vec3> velocities(times.size()), positions(times.size());
    for (std::size_t k = 0; k < times.size(); ++k) {
        const std::size_t j = std::min<std::size_t>(static_cast<std::size_t>(std::upper_bound(key_times.begin(), key_times.end(), times[k]) - key_times.begin()), key_times.size() - 1);
        const std::size_t i = j == 0 ? 0 : j - 1;
        const double s = j == i ? 0.0 : (times[k] - key_times[i]) / (key_times[j] - key_times[i]);
        rotations[k] = key_rotations[i].slerp(s, key_rotations[j]).toRotationMatrix();
        velocities[k] = spline.derivative(times[k]);
    }
    positions[0] = spline.value(times[0]);
    const Vec3 gravity(0.0, 0.0, -9.81);

    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    auto gaussian = [&]() { return Vec3(normal(rng), normal(rng), normal(rng)); };
    Vec3 accelerometer_bias = Vec3::Zero(), gyroscope_bias = Vec3::Zero();

    sequence.imu.push_back({times[0], Vec3::Zero(), rotations[0].transpose() * (-gravity)});  // static start
    sequence.ground_truth.emplace_back(Pose(rotations[0], positions[0]));
    for (std::size_t k = 0; k + 1 < times.size(); ++k) {
        const double dt = times[k + 1] - times[k];
        const Vec3 angular_velocity = logSO3(rotations[k].transpose() * rotations[k + 1]) / dt;
        const Vec3 acceleration = (velocities[k + 1] - velocities[k]) / dt;
        positions[k + 1] = positions[k] + 0.5 * (velocities[k] + velocities[k + 1]) * dt;
        const Vec3 specific_force = rotations[k].transpose() * (acceleration - gravity);

        // Continuous densities -> per-sample values: white noise sigma / sqrt(dt), bias random walk sigma * sqrt(dt)
        accelerometer_bias += noise.accelerometer_random_walk * std::sqrt(dt) * gaussian();
        gyroscope_bias += noise.gyroscope_random_walk * std::sqrt(dt) * gaussian();
        ImuMeasurement m;
        m.timestamp = times[k + 1];
        m.linear_acceleration = specific_force + accelerometer_bias + noise.accelerometer_noise_density / std::sqrt(dt) * gaussian();
        m.angular_velocity = angular_velocity + gyroscope_bias + noise.gyroscope_noise_density / std::sqrt(dt) * gaussian();
        sequence.imu.push_back(m);
        sequence.ground_truth.emplace_back(Pose(rotations[k + 1], positions[k + 1]));
    }
    return sequence;
}

}  // namespace msckf
