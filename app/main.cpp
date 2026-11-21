// msckf_run: runs the visual-inertial MSCKF on a dataset sequence, evaluates it against the ground truth and
// optionally streams it to Rerun.
//
//   msckf_run --sequence V1_01_easy
//   msckf_run --sequence V1_02_medium --loop_closure --save_results

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <numeric>

#include <CLI/CLI.hpp>
#include <opencv2/imgproc.hpp>

#include "dataset.hpp"
#include "evaluation.hpp"
#include "msckf/vio_system.hpp"
#include "msckf_frontend/xfeat.hpp"
#include "visualizer.hpp"

using namespace msckf;

namespace {

struct Options {
    std::string data = "./data";
    std::string source = "euroc";
    std::string sequence = "V1_01_easy";
    std::string model = "models/xfeat.onnx";
    long long max_frames = 1'000'000;
    bool save_results = false;
    bool no_viewer = false;
    bool only_imu = false;
    bool loop_closure = false;
    bool coreml = false;
    int top_k = 300;
    double sigma_image = 2.0;
    double max_triangulation_error = 6.0;
    double zupt_max_disparity = 1.0;
    double epipolar_threshold = 10.0;
    double init_sigma_theta = 0.05;
    double init_sigma_ba = 0.5;
    double imu_noise_scale = 10.0;
    double imu_walk_scale = 1.0;
    double min_parallax = 45.0;
    int max_track_length = 10;
    bool no_update_on_prune = false;
    LoopClosureParameters loop;
};

std::string formatTime(const std::vector<double>& times) {
    const double mean = times.empty() ? 0.0 : std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    char buffer[128];
    std::snprintf(buffer, sizeof(buffer), "%.8f s - %.8f Hz", mean, mean > 0 ? 1.0 / mean : 0.0);
    return buffer;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    CLI::App app{"Monocular visual-inertial MSCKF"};
    app.add_option("--data", o.data, "Dataset root folder");
    app.add_option("--source", o.source, "Dataset source")->check(CLI::IsMember({"euroc"}));
    app.add_option("--sequence", o.sequence, "Sequence name (V1_01_easy, MH_01_easy, ...)");
    app.add_option("--model", o.model, "XFeat ONNX model");
    app.add_option("--max_frames", o.max_frames, "Maximum number of IMU samples to process");
    app.add_flag("--save_results", o.save_results, "Save results, trajectories and the Rerun recording");
    app.add_flag("--no_viewer", o.no_viewer, "Do not stream to Rerun");
    app.add_flag("--only_imu", o.only_imu, "Propagate with the IMU only (diagnostics)");
    app.add_flag("--loop_closure", o.loop_closure, "Enable loop closure on keyframes");
    app.add_flag("--coreml", o.coreml, "Run XFeat with CoreML (Apple GPU / Neural Engine)");
    app.add_option("--top_k", o.top_k, "XFeat keypoints per image");
    app.add_option("--sigma_image", o.sigma_image, "Standard deviation of a keypoint measurement [px]");
    app.add_option("--max_triangulation_error", o.max_triangulation_error, "Maximum RMS reprojection error of a triangulated feature [px]");
    app.add_option("--zupt_max_disparity", o.zupt_max_disparity, "Zero-velocity update below this median feature displacement [px]");
    app.add_option("--epipolar_threshold", o.epipolar_threshold, "Sampson distance above which a match is rejected [px]");
    app.add_option("--init_sigma_theta", o.init_sigma_theta, "Initial orientation standard deviation [rad]");
    app.add_option("--init_sigma_ba", o.init_sigma_ba, "Initial accelerometer bias standard deviation [m/s^2]");
    app.add_option("--imu_noise_scale", o.imu_noise_scale, "Multiplier of the datasheet IMU white-noise densities");
    app.add_option("--imu_walk_scale", o.imu_walk_scale, "Multiplier of the datasheet IMU bias random-walk densities");
    app.add_option("--min_parallax", o.min_parallax, "Parallax that triggers the use of a tracked feature [deg]");
    app.add_option("--max_track_length", o.max_track_length, "Also use tracked features with this many observations (0 disables it)");
    app.add_flag("--no_update_on_prune", o.no_update_on_prune, "Do not use the features of the pruned clones");
    app.add_option("--lc_descriptor_similarity", o.loop.min_descriptor_similarity, "Loop closure: minimum cosine similarity of a descriptor match");
    app.add_option("--lc_min_inliers", o.loop.min_inliers, "Loop closure: PnP inliers to accept a loop");
    app.add_option("--lc_pnp_threshold", o.loop.pnp_threshold, "Loop closure: reprojection error of a PnP inlier [px]");
    app.add_option("--lc_recent_excluded", o.loop.recent_keyframes_excluded, "Loop closure: candidates must be older than this many keyframes");
    CLI11_PARSE(app, argc, argv);

    //* Sequence
    Sequence sequence = loadEuroc(o.data, o.sequence);
    // Datasheet values underestimate the noise of a real IMU
    ImuNoise noise = sequence.imu_noise;
    noise.accelerometer_noise_density *= o.imu_noise_scale;
    noise.gyroscope_noise_density *= o.imu_noise_scale;
    noise.accelerometer_random_walk *= o.imu_walk_scale;
    noise.gyroscope_random_walk *= o.imu_walk_scale;
    std::printf("Running VIO on %s %s: %zu IMU samples, %zu images\n", o.source.c_str(), o.sequence.c_str(), sequence.imu.size(), sequence.images.size());

    //* Filter
    Parameters p;
    p.K = sequence.K;
    p.T_I_C = sequence.T_I_C;
    p.sigma_image = o.sigma_image;
    p.only_imu = o.only_imu;
    p.accelerometer_noise_density = noise.accelerometer_noise_density;
    p.gyroscope_noise_density = noise.gyroscope_noise_density;
    p.accelerometer_random_walk = noise.accelerometer_random_walk;
    p.gyroscope_random_walk = noise.gyroscope_random_walk;
    p.gravity = Vec3(0.0, 0.0, -9.81);
    // Real IMU biases are not zero: initial gyroscope bias from the static start, uncertain initial state
    p.estimate_initial_gyroscope_bias = true;
    p.initial_covariance_diagonal << Vec3::Constant(o.init_sigma_theta * o.init_sigma_theta), Vec3::Constant(0.005 * 0.005),
        Vec3::Constant(0.01 * 0.01), Vec3::Constant(o.init_sigma_ba * o.init_sigma_ba), Vec3::Zero();
    p.min_cosine_similarity = 0.95;
    p.use_parallax = true;
    p.min_parallax = o.min_parallax;
    p.epipolar_rejection_threshold = o.epipolar_threshold;
    p.homography_rejection_threshold = 5.0;
    p.min_number_of_frames_to_be_tracked = 4;
    p.min_number_of_frames_to_be_lost = 2;
    p.max_number_of_camera_states = 30;
    p.max_triangulation_error = o.max_triangulation_error;
    p.zupt_max_disparity = o.zupt_max_disparity;
    p.max_track_length = o.max_track_length;
    p.update_features_on_prune = !o.no_update_on_prune;
    VioSystem system(p, o.loop, o.loop_closure);

    XFeatExtractor::Options extractor_options;
    extractor_options.top_k = o.top_k;
    extractor_options.use_coreml = o.coreml;
    const XFeatExtractor extractor(o.model, extractor_options);

    const std::string results_folder = o.data + "/" + o.source + "/" + o.sequence + "/results";
    if (o.save_results) std::filesystem::create_directories(results_folder);
    Visualizer visualizer(sequence, o.no_viewer ? std::nullopt : std::optional<std::string>(o.save_results ? results_folder + "/recording.rrd" : ""));

    //* Front-end in a separate thread: the next image is read, undistorted and passed to XFeat while the filter
    //* processes the current one (the time spent waiting for it is counted in the loop time)
    auto load = [&](std::size_t index) {
        cv::Mat image = sequence.loadImage(index), gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        return std::make_pair(image, extractor.extract(gray));
    };
    std::size_t camera_index = 1;  // the first image only marks the end of the static initialization window
    std::future<std::pair<cv::Mat, FrameFeatures>> next_frame = std::async(std::launch::async, load, camera_index);

    Evaluator evaluator;
    std::vector<double> loop_time, loop_time_camera, loop_time_imu;
    std::ofstream trajectory;
    if (o.save_results) {
        trajectory.open(results_folder + "/trajectory.csv");
        trajectory << "timestamp,x,y,z,qw,qx,qy,qz,gt_x,gt_y,gt_z,corrected_x,corrected_y,corrected_z\n";
    }

    const std::size_t steps = std::min<std::size_t>(sequence.imu.size(), static_cast<std::size_t>(o.max_frames));
    double first_timestamp = sequence.imu.front().timestamp, last_timestamp = first_timestamp;
    for (std::size_t i = 0; i < steps; ++i) {
        if (camera_index + 1 > sequence.images.size() - 1) break;
        const ImuMeasurement& measurement = sequence.imu[i];
        const auto start = std::chrono::steady_clock::now();

        system.addImu(measurement);
        bool camera = false;
        cv::Mat image;
        if (measurement.timestamp >= sequence.images[camera_index].timestamp - 1e-6) {
            camera = true;
            auto [frame_image, features] = next_frame.get();
            image = frame_image;
            const double timestamp = sequence.images[camera_index].timestamp;
            if (++camera_index < sequence.images.size()) next_frame = std::async(std::launch::async, load, camera_index);
            system.addImage(timestamp, features);
        }

        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        loop_time.push_back(elapsed);
        (camera ? loop_time_camera : loop_time_imu).push_back(elapsed);
        last_timestamp = measurement.timestamp;

        //* Evaluation and output
        const Msckf& filter = system.filter();
        const Pose estimate = system.odometryPose();
        const std::optional<Pose> corrected = o.loop_closure ? std::optional<Pose>(system.correctedPose()) : std::nullopt;
        const bool evaluated = evaluator.add(estimate, filter.covariance().block<3, 3>(12, 12), filter.covariance().block<3, 3>(0, 0),
                                             sequence.ground_truth[i], corrected);
        visualizer.step(static_cast<long long>(i), measurement.timestamp, system, evaluated ? evaluator.alignedGroundTruth() : std::nullopt,
                        camera ? &image : nullptr, evaluator);
        if (o.save_results) {
            const Eigen::Quaterniond q(estimate.R);
            const Pose c = corrected.value_or(estimate);
            const Vec3 gt = evaluated ? evaluator.alignedGroundTruth()->t : Vec3::Constant(std::nan(""));
            trajectory << measurement.timestamp << ',' << estimate.t.x() << ',' << estimate.t.y() << ',' << estimate.t.z() << ',' << q.w() << ','
                       << q.x() << ',' << q.y() << ',' << q.z() << ',' << gt.x() << ',' << gt.y() << ',' << gt.z() << ',' << c.t.x() << ','
                       << c.t.y() << ',' << c.t.z() << '\n';
        }
    }

    //* Summary (same format as the Python implementation, parsed by the benchmark)
    const Msckf& filter = system.filter();
    char buffer[2048];
    std::string summary = "VIO:\n" + evaluator.summary();
    std::snprintf(buffer, sizeof(buffer),
                  " - Final IMU biases:\n   - Gyroscope:      [%.4f %.4f %.4f] rad/s\n   - Accelerometer:  [%.3f %.3f %.3f] m/s^2\n"
                  " - Rejections:\n   - Epipolar:       %lld\n   - Homography:     %lld\n   - Gating (χ²):    %lld\n   - Triangulation:  %lld\n"
                  " - Zero-velocity updates: %lld\n",
                  filter.imu().gyroscope_bias.x(), filter.imu().gyroscope_bias.y(), filter.imu().gyroscope_bias.z(), filter.imu().accelerometer_bias.x(),
                  filter.imu().accelerometer_bias.y(), filter.imu().accelerometer_bias.z(), filter.epipolarRejections(), filter.homographyRejections(),
                  filter.gatingRejections(), filter.triangulationRejections(), filter.zeroVelocityUpdates());
    summary += buffer;
    if (system.loopClosure()) {
        const LoopClosureStatistics& s = system.loopClosure()->statistics();
        std::snprintf(buffer, sizeof(buffer),
                      " - Loop closure: %zu keyframes, %zu loops\n   - Mean 3D points per keyframe: %.1f\n   - Best global similarity: %.3f\n"
                      "   - Discarded: %lld without landmarks, %lld below global similarity, %lld too few matches, %lld PnP, %lld implausible\n",
                      system.loopClosure()->graph().nodes().size(), system.loopClosure()->loops().size(), s.mean_landmarks, s.best_global_similarity,
                      s.keyframes_without_landmarks, s.below_global_similarity, s.too_few_matches, s.pnp_failed, s.implausible_correction);
        summary += buffer;
    }
    const double total = std::accumulate(loop_time.begin(), loop_time.end(), 0.0);
    std::snprintf(buffer, sizeof(buffer), " - Real-time factor: %.2f (processing time / data duration, < 1 is faster than real time)\n",
                  total / std::max(last_timestamp - first_timestamp, 1e-9));
    summary += buffer;
    summary += " - Profiling:\n   - Mean:           " + formatTime(loop_time) + "\n   - With Camera:    " + formatTime(loop_time_camera) +
               "\n   - Without Camera: " + formatTime(loop_time_imu) + "\n";
    std::cout << summary;
    if (o.save_results) std::ofstream(results_folder + "/results.txt") << summary;
    return 0;
}
