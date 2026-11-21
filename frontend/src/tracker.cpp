#include "msckf_frontend/tracker.hpp"

#include <opencv2/calib3d.hpp>

namespace msckf {

namespace {

/// First index of the maximum.
template <typename Vector>
Eigen::Index argmax(const Vector& v) {
    Eigen::Index best = 0;
    for (Eigen::Index i = 1; i < v.size(); ++i)
        if (v(i) > v(best)) best = i;
    return best;
}

}  // namespace

Tracker::Result Tracker::track(const FrameFeatures& features) {
    Result result;

    // Keep keypoints with at least a fraction of the mean detection score
    double score_mean = 0.0;
    for (float s : features.scores) score_mean += s;
    score_mean /= std::max<std::size_t>(features.scores.size(), 1);
    std::vector<int> kept;
    for (std::size_t i = 0; i < features.scores.size(); ++i)
        if (features.scores[i] >= options_.min_score_ratio * score_mean) kept.push_back(static_cast<int>(i));

    //* Mutual nearest neighbours between the track descriptors (score-weighted mean) and the current descriptors
    std::vector<long long> track_ids;
    std::vector<std::pair<long long, int>> matches;  // (track id, keypoint index)
    std::vector<bool> keypoint_matched(features.scores.size(), false);
    if (!tracks_.empty() && !kept.empty()) {
        const Eigen::Index size = features.descriptors.cols();
        Eigen::MatrixXf track_descriptors(static_cast<Eigen::Index>(tracks_.size()), size);
        for (const auto& [id, track] : tracks_) {
            Eigen::VectorXf weighted = Eigen::VectorXf::Zero(size);
            float weight_sum = 0.0f;
            for (std::size_t k = 0; k < track.descriptors.size(); ++k) {
                weighted += track.scores[k] * track.descriptors[k];
                weight_sum += track.scores[k];
            }
            track_descriptors.row(static_cast<Eigen::Index>(track_ids.size())) = (weighted / weight_sum).transpose();
            track_ids.push_back(id);
        }
        Eigen::MatrixXf current(static_cast<Eigen::Index>(kept.size()), size);
        for (std::size_t j = 0; j < kept.size(); ++j) current.row(static_cast<Eigen::Index>(j)) = features.descriptors.row(kept[j]);

        const Eigen::MatrixXf similarity = track_descriptors * current.transpose();
        std::vector<Eigen::Index> best_current(similarity.rows()), best_track(similarity.cols());
        for (Eigen::Index i = 0; i < similarity.rows(); ++i) best_current[i] = argmax(similarity.row(i));
        for (Eigen::Index j = 0; j < similarity.cols(); ++j) best_track[j] = argmax(similarity.col(j));
        for (Eigen::Index i = 0; i < similarity.rows(); ++i) {
            const Eigen::Index j = best_current[i];
            if (best_track[j] == i && similarity(i, j) > options_.min_cosine_similarity) matches.emplace_back(track_ids[i], kept[j]);
        }
    }

    //* Geometric verification: RANSAC fundamental matrix between the previous and the current position of the tracks
    std::vector<bool> inlier(matches.size(), true);
    if (matches.size() >= 8) {
        std::vector<cv::Point2f> previous, current;
        for (const auto& [id, j] : matches) {
            previous.emplace_back(static_cast<float>(tracks_.at(id).last_keypoint.x()), static_cast<float>(tracks_.at(id).last_keypoint.y()));
            current.emplace_back(static_cast<float>(features.keypoints[j].x()), static_cast<float>(features.keypoints[j].y()));
        }
        std::vector<uchar> mask;
        const cv::Mat F = cv::findFundamentalMat(previous, current, cv::FM_RANSAC, options_.ransac_threshold, options_.ransac_confidence, mask);
        if (!F.empty() && mask.size() == matches.size())
            for (std::size_t k = 0; k < matches.size(); ++k) inlier[k] = mask[k] != 0;
    }

    //* Observations: verified matches continue their tracks, unmatched keypoints start new ones
    for (std::size_t k = 0; k < matches.size(); ++k) {
        const auto [id, j] = matches[k];
        keypoint_matched[j] = true;  // a rejected match is dropped, not turned into a new track
        if (!inlier[k]) {
            result.rejected.push_back(features.keypoints[j]);
            ++rejected_matches_;
            continue;
        }
        Track& track = tracks_.at(id);
        track.descriptors.push_back(features.descriptors.row(j).transpose());
        track.scores.push_back(features.scores[j]);
        if (static_cast<int>(track.descriptors.size()) > options_.max_descriptors_per_track) {
            track.descriptors.erase(track.descriptors.begin());
            track.scores.erase(track.scores.begin());
        }
        track.last_keypoint = features.keypoints[j];
        track.lost_frames = -1;  // reset below
        result.frame.observations.push_back({id, features.keypoints[j], features.descriptors.row(j).transpose(), features.scores[j]});
        ++result.matched;
    }
    for (auto it = tracks_.begin(); it != tracks_.end();) {
        if (++it->second.lost_frames >= options_.max_lost_frames) it = tracks_.erase(it);
        else ++it;
    }
    for (int j : kept) {
        if (keypoint_matched[j]) continue;
        const long long id = next_id_++;
        Track& track = tracks_[id];
        track.descriptors.push_back(features.descriptors.row(j).transpose());
        track.scores.push_back(features.scores[j]);
        track.last_keypoint = features.keypoints[j];
        result.frame.observations.push_back({id, features.keypoints[j], features.descriptors.row(j).transpose(), features.scores[j]});
    }
    return result;
}

}  // namespace msckf
