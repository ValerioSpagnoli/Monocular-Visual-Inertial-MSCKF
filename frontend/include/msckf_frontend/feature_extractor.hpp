#pragma once

#include <opencv2/core.hpp>

#include "msckf/msckf.hpp"

namespace msckf {

/// Keypoints and descriptors of an image for the backend: keypoints [px], one L2-normalized float descriptor per row
/// (the backend matches them by cosine similarity) and detection scores.
class FeatureExtractor {
public:
    virtual ~FeatureExtractor() = default;
    virtual FrameFeatures extract(const cv::Mat& grayscale) const = 0;
};

}  // namespace msckf
