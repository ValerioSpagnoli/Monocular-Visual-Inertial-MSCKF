#include "dataset.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

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

}  // namespace msckf
