#pragma once

#include "msckf_frontend/feature_extractor.hpp"

namespace cv {
class Feature2D;  // features2d (OpenCV 4) or features (OpenCV 5)
}

namespace msckf {

/// Classical (non-learned) keypoints and descriptors from OpenCV, for comparison with the networks.
/// Keypoints are spread over the image with a grid (at most top_k / cells per cell first, then the strongest of the
/// rest), as a VIO frontend needs coverage more than the strongest corners. Descriptors are converted to
/// L2-normalized float vectors so that the cosine similarity used by the backend is equivalent to the native distance:
/// binary descriptors (ORB, AKAZE) become ±1/√bits (cosine = 1 - 2 Hamming / bits), SIFT becomes RootSIFT.
class OpenCvExtractor : public FeatureExtractor {
public:
    enum class Type { Orb, Sift, Akaze };

    struct Options {
        Type type = Type::Orb;
        int top_k = 300;
        int grid_cols = 8;
        int grid_rows = 6;
    };

    explicit OpenCvExtractor(const Options& options);

    FrameFeatures extract(const cv::Mat& grayscale) const override;

private:
    Options options_;
    cv::Ptr<cv::Feature2D> detector_;
};

}  // namespace msckf
