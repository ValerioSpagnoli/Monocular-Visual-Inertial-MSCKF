#include "visualizer.hpp"

#include <algorithm>
#include <cstdio>
#include <set>

#include <opencv2/imgcodecs.hpp>
#include <rerun.hpp>

namespace msckf {

namespace {

const rerun::Color kGroundTruth(40, 190, 80), kEstimate(40, 130, 255), kCorrected(255, 120, 40);
const rerun::Color kTrack(40, 220, 120), kNew(80, 200, 255), kUpdate(255, 160, 0), kGated(200, 0, 200), kRejected(255, 50, 50);
const rerun::Color kMap(200, 200, 200), kWindow(255, 200, 60), kCovariance(255, 80, 80);
constexpr const char* kCamera = "world/est/body/camera";

rerun::Position3D position(const Vec3& p) { return {static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z())}; }

rerun::Transform3D transform(const Pose& T) {
    std::array<float, 9> columns;
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r) columns[static_cast<std::size_t>(3 * c + r)] = static_cast<float>(T.R(r, c));
    return rerun::Transform3D::from_translation_mat3x3(rerun::components::Translation3D(static_cast<float>(T.t.x()), static_cast<float>(T.t.y()),
                                                                                        static_cast<float>(T.t.z())),
                                                       rerun::components::TransformMat3x3(columns));
}

rerun::LineStrip3D strip(const std::vector<std::array<float, 3>>& points) {
    std::vector<rerun::Vec3D> vertices;
    vertices.reserve(points.size());
    for (const auto& p : points) vertices.emplace_back(p[0], p[1], p[2]);
    return rerun::LineStrip3D(vertices);
}

}  // namespace

Visualizer::Visualizer(const Sequence& sequence, const std::optional<std::string>& destination) {
    if (!destination) return;
    rec_ = std::make_unique<rerun::RecordingStream>("vio");
    if (destination->empty()) {
        rec_->spawn().exit_on_failure();
    } else {
        // Viewer and .rrd file at the same time
        rerun::spawn().exit_on_failure();
        rec_->set_sinks(rerun::GrpcSink{}, rerun::FileSink{*destination}).exit_on_failure();
    }

    rec_->log_static("world", rerun::ViewCoordinates::RIGHT_HAND_Z_UP);
    rec_->log_static("world/est/body", rerun::TransformAxes3D(0.3f));
    rec_->log_static("world/gt/body", rerun::TransformAxes3D(0.3f));
    rec_->log_static(kCamera, transform(sequence.T_I_C));
    std::array<float, 9> K;
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r) K[static_cast<std::size_t>(3 * c + r)] = static_cast<float>(sequence.K(r, c));
    rec_->log_static(kCamera, rerun::Pinhole(rerun::components::PinholeProjection(K))
                                  .with_resolution(sequence.width, sequence.height)
                                  .with_camera_xyz(rerun::components::ViewCoordinates::RDF)
                                  .with_image_plane_distance(0.4f)
                                  .with_color(kEstimate));
}

Visualizer::~Visualizer() = default;

void Visualizer::step(long long index, double timestamp, const VioSystem& system, const std::optional<Pose>& ground_truth, const cv::Mat* image,
                      const std::vector<Vec2>& rejected, const Evaluator& evaluator) {
    if (!rec_) return;
    rec_->set_time_sequence("step", index);
    rec_->set_time_duration_secs("time", timestamp);
    rec_->log("world/est/body", transform(system.odometryPose()));
    if (ground_truth) {
        rec_->log("world/gt/body", transform(*ground_truth));
        last_ground_truth_ = ground_truth;
    }
    if (image) logFrame(system, *image, rejected, timestamp, evaluator);
}

void Visualizer::logFrame(const VioSystem& system, const cv::Mat& image, const std::vector<Vec2>& rejected, double timestamp, const Evaluator& evaluator) {
    const Msckf& filter = system.filter();
    ++frames_;

    //* Image and features (colored by status) with their recent tracks
    std::vector<uint8_t> jpeg;
    cv::imencode(".jpg", image, jpeg, {cv::IMWRITE_JPEG_QUALITY, 75});
    rec_->log(std::string(kCamera) + "/image", rerun::EncodedImage::from_bytes(jpeg, rerun::components::MediaType::jpeg()));

    const std::set<long long> updated(filter.updatedFeatureIds().begin(), filter.updatedFeatureIds().end());
    const std::set<long long> gated(filter.gatedFeatureIds().begin(), filter.gatedFeatureIds().end());
    std::vector<rerun::Vec2D> points;
    std::vector<rerun::Color> colors, track_colors;
    std::vector<float> radii;
    std::vector<rerun::LineStrip2D> tracks;
    int visible = 0, tracked = 0;
    for (const auto& [id, feature] : filter.features()) {
        if (feature.camera_ids.empty() || feature.camera_ids.back() != filter.imu().id) continue;
        ++visible;
        rerun::Color color = kNew;
        float radius = 2.0f;
        if (updated.count(id)) { color = kUpdate; radius = 4.0f; }
        else if (gated.count(id)) { color = kGated; radius = 4.0f; }
        else if (feature.keypoints.size() > 1) { color = kTrack; radius = 2.5f; }
        if (feature.keypoints.size() > 1) ++tracked;
        points.emplace_back(static_cast<float>(feature.keypoints.back().x()), static_cast<float>(feature.keypoints.back().y()));
        colors.push_back(color);
        radii.push_back(radius);
        if (feature.keypoints.size() > 1) {
            std::vector<rerun::Vec2D> track;
            for (std::size_t k = feature.keypoints.size() > 10 ? feature.keypoints.size() - 10 : 0; k < feature.keypoints.size(); ++k)
                track.emplace_back(static_cast<float>(feature.keypoints[k].x()), static_cast<float>(feature.keypoints[k].y()));
            tracks.emplace_back(track);
            track_colors.push_back(color);
        }
    }
    rec_->log(std::string(kCamera) + "/features", rerun::Points2D(points).with_colors(colors).with_radii(radii));
    rec_->log(std::string(kCamera) + "/tracks", rerun::LineStrips2D(tracks).with_colors(track_colors).with_radii(0.8f));
    std::vector<rerun::Vec2D> rejected_points;
    for (const Vec2& r : rejected) rejected_points.emplace_back(static_cast<float>(r.x()), static_cast<float>(r.y()));
    rec_->log(std::string(kCamera) + "/rejected", rerun::Points2D(rejected_points).with_colors(kRejected).with_radii(4.0f));

    //* Sliding window, landmarks of the current update, 3-sigma position ellipsoid
    std::vector<rerun::Position3D> window;
    for (const auto& [id, clone] : filter.cameras()) window.push_back(position(clone.T_W_C.t));
    rec_->log("world/est/sliding_window", rerun::Points3D(window).with_colors(kWindow).with_radii(0.03f));
    std::vector<rerun::Position3D> current;
    for (long long id : filter.updatedFeatureIds()) {
        const auto it = filter.triangulatedPoints().find(id);
        if (it != filter.triangulatedPoints().end()) current.push_back(position(it->second));
    }
    rec_->log("world/map/current", rerun::Points3D(current).with_colors(kUpdate).with_radii(0.04f));
    const Pose estimate = system.odometryPose();
    const Vec3 sigma = filter.covariance().block<3, 3>(12, 12).diagonal().cwiseMax(0.0).cwiseSqrt();
    rec_->log("world/est/position_3sigma", rerun::Ellipsoids3D::from_centers_and_half_sizes(
                                               {position(estimate.t)}, {rerun::HalfSize3D(3 * static_cast<float>(sigma.x()), 3 * static_cast<float>(sigma.y()),
                                                                                           3 * static_cast<float>(sigma.z()))})
                                               .with_colors(kCovariance)
                                               .with_fill_mode(rerun::components::FillMode::MajorWireframe));

    //* Trajectories and map, re-logged as a whole every 5 images
    estimate_trajectory_.push_back({static_cast<float>(estimate.t.x()), static_cast<float>(estimate.t.y()), static_cast<float>(estimate.t.z())});
    if (last_ground_truth_) ground_truth_trajectory_.push_back({static_cast<float>(last_ground_truth_->t.x()), static_cast<float>(last_ground_truth_->t.y()),
                                                                static_cast<float>(last_ground_truth_->t.z())});
    if (system.loopClosure()) {
        const Pose corrected = system.correctedPose();
        corrected_trajectory_.push_back({static_cast<float>(corrected.t.x()), static_cast<float>(corrected.t.y()), static_cast<float>(corrected.t.z())});
    }
    for (const auto& [id, point] : filter.triangulatedPoints()) landmarks_[id] = point;
    if (frames_ % 5 == 0) {
        rec_->log("world/est/trajectory", rerun::LineStrips3D(strip(estimate_trajectory_)).with_colors(kEstimate).with_radii(0.01f));
        if (ground_truth_trajectory_.size() > 1) rec_->log("world/gt/trajectory", rerun::LineStrips3D(strip(ground_truth_trajectory_)).with_colors(kGroundTruth).with_radii(0.01f));
        if (corrected_trajectory_.size() > 1) rec_->log("world/corrected/trajectory", rerun::LineStrips3D(strip(corrected_trajectory_)).with_colors(kCorrected).with_radii(0.01f));
        std::vector<rerun::Position3D> map;
        for (const auto& [id, point] : landmarks_)
            if ((point - estimate.t).norm() < 30.0) map.push_back(position(point));  // far points come from low-parallax features
        rec_->log("world/map/landmarks", rerun::Points3D(map).with_colors(kMap).with_radii(0.015f));
    }

    //* Text panels
    rec_->log("text/metrics", rerun::TextDocument(evaluator.markdown()).with_media_type(rerun::components::MediaType::markdown()));
    char buffer[1536];
    const Vec3& bg = filter.imu().gyroscope_bias;
    const Vec3& ba = filter.imu().accelerometer_bias;
    std::snprintf(buffer, sizeof(buffer),
                  "| Filter | Value |\n|---|---|\n| Time | %.2f s |\n| Features in image (tracked / new) | %d (%d / %d) |\n"
                  "| Features in the filter | %zu |\n| Rejected matches | %zu |\n| Features in last update (used / gated) | %zu / %zu |\n"
                  "| Camera states in window | %zu |\n| Gyroscope bias | [%+.4f, %+.4f, %+.4f] rad/s |\n| Accelerometer bias | [%+.3f, %+.3f, %+.3f] m/s² |\n"
                  "| Zero-velocity updates | %lld |\n| Loops closed | %zu |\n\n"
                  "**Image legend:** orange = used in update, green = tracked, light blue = new, magenta = gated, red = match rejected by RANSAC",
                  timestamp, visible, tracked, visible - tracked, filter.features().size(), rejected.size(), filter.updatedFeatureIds().size(),
                  filter.gatedFeatureIds().size(), filter.cameras().size(), bg.x(), bg.y(), bg.z(), ba.x(), ba.y(), ba.z(), filter.zeroVelocityUpdates(),
                  system.loopClosure() ? system.loopClosure()->loops().size() : std::size_t{0});
    rec_->log("text/status", rerun::TextDocument(buffer).with_media_type(rerun::components::MediaType::markdown()));
}

}  // namespace msckf
