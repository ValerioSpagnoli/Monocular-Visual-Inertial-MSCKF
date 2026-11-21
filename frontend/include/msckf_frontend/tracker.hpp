#pragma once

#include <map>
#include <vector>

#include "msckf/msckf.hpp"
#include "msckf_frontend/xfeat.hpp"

namespace msckf {

/// Feature tracking across images: descriptor matching against the active tracks and geometric verification.
/// It does not need the filter state: wrong matches are found with a RANSAC fundamental matrix between the previous
/// and the current position of the tracks (as in VINS-Mono and OpenVINS).
class Tracker {
public:
    struct Options {
        float min_cosine_similarity = 0.95f;  // mutual nearest neighbours above this similarity
        float min_score_ratio = 0.5f;         // keep keypoints with at least this fraction of the mean detection score
        double ransac_threshold = 1.5;        // distance from the epipolar line [px]
        double ransac_confidence = 0.99;
        int max_lost_frames = 2;              // a track not matched in this many consecutive images ends
        int max_descriptors_per_track = 20;   // the track descriptor is the score-weighted mean of the last ones
    };

    struct Result {
        TrackedFrame frame;                   // observations of tracked and new features
        std::vector<Vec2> rejected;           // current keypoints of the matches discarded by RANSAC (diagnostics)
        int matched = 0;                      // observations of existing tracks
    };

    explicit Tracker(const Options& options) : options_(options) {}

    Result track(const FrameFeatures& features);

    long long rejectedMatches() const { return rejected_matches_; }
    std::size_t activeTracks() const { return tracks_.size(); }

private:
    struct Track {
        std::vector<Eigen::VectorXf> descriptors;
        std::vector<float> scores;
        Vec2 last_keypoint = Vec2::Zero();
        int lost_frames = 0;
    };

    Options options_;
    std::map<long long, Track> tracks_;
    long long next_id_ = 1;
    long long rejected_matches_ = 0;
};

}  // namespace msckf
