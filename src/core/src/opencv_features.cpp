#include "msckf/opencv_features.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#if CV_VERSION_MAJOR >= 5
#include <opencv2/features.hpp>
#include <opencv2/xfeatures2d.hpp>  // AKAZE moved to opencv_contrib in OpenCV 5
#else
#include <opencv2/features2d.hpp>
#endif

namespace msckf {

namespace {

cv::Ptr<cv::Feature2D> createDetector(OpenCvExtractor::Type type, int top_k) {
    switch (type) {
        // More candidates than needed, the grid selects among them
        case OpenCvExtractor::Type::Orb: return cv::ORB::create(4 * top_k, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 10);
        case OpenCvExtractor::Type::Sift: return cv::SIFT::create(0, 3, 0.02);
#if CV_VERSION_MAJOR >= 5
        case OpenCvExtractor::Type::Akaze: return cv::xfeatures2d::AKAZE::create(cv::xfeatures2d::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.0005f);
#else
        case OpenCvExtractor::Type::Akaze: return cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.0005f);
#endif
    }
    return nullptr;
}

/// Strongest keypoints spread over a grid: first up to ceil(top_k / cells) per cell, then the strongest remaining.
std::vector<cv::KeyPoint> selectOnGrid(std::vector<cv::KeyPoint> keypoints, const cv::Size& size, int top_k, int cols, int rows) {
    std::stable_sort(keypoints.begin(), keypoints.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) { return a.response > b.response; });
    const int per_cell = (top_k + cols * rows - 1) / (cols * rows);
    std::vector<int> count(static_cast<std::size_t>(cols * rows), 0);
    std::vector<bool> taken(keypoints.size(), false);
    std::vector<cv::KeyPoint> selected;
    for (std::size_t k = 0; k < keypoints.size() && static_cast<int>(selected.size()) < top_k; ++k) {
        const int cx = std::clamp(static_cast<int>(keypoints[k].pt.x * cols / size.width), 0, cols - 1);
        const int cy = std::clamp(static_cast<int>(keypoints[k].pt.y * rows / size.height), 0, rows - 1);
        if (count[static_cast<std::size_t>(cy * cols + cx)]++ < per_cell) {
            selected.push_back(keypoints[k]);
            taken[k] = true;
        }
    }
    for (std::size_t k = 0; k < keypoints.size() && static_cast<int>(selected.size()) < top_k; ++k)
        if (!taken[k]) selected.push_back(keypoints[k]);
    return selected;
}

}  // namespace

OpenCvExtractor::OpenCvExtractor(const Options& options) : options_(options), detector_(createDetector(options.type, options.top_k)) {}

FrameFeatures OpenCvExtractor::extract(const cv::Mat& grayscale) const {
    std::vector<cv::KeyPoint> keypoints;
    detector_->detect(grayscale, keypoints);
    keypoints = selectOnGrid(std::move(keypoints), grayscale.size(), options_.top_k, options_.grid_cols, options_.grid_rows);
    cv::Mat descriptors;
    detector_->compute(grayscale, keypoints, descriptors);  // may drop keypoints near the border

    FrameFeatures features;
    features.descriptors.resize(descriptors.rows, descriptors.depth() == CV_8U ? descriptors.cols * 8 : descriptors.cols);
    for (int r = 0; r < descriptors.rows; ++r) {
        Eigen::RowVectorXf d(features.descriptors.cols());
        if (descriptors.depth() == CV_8U) {
            const uchar* bytes = descriptors.ptr<uchar>(r);
            for (int b = 0; b < descriptors.cols * 8; ++b) d(b) = (bytes[b / 8] >> (b % 8)) & 1 ? 1.0f : -1.0f;
        } else {
            const float* values = descriptors.ptr<float>(r);
            for (int c = 0; c < descriptors.cols; ++c) d(c) = values[c];
            d = (d / std::max(d.lpNorm<1>(), 1e-12f)).cwiseSqrt();  // RootSIFT
        }
        features.descriptors.row(r) = d / std::max(d.norm(), 1e-12f);
        features.keypoints.emplace_back(keypoints[static_cast<std::size_t>(r)].pt.x, keypoints[static_cast<std::size_t>(r)].pt.y);
        features.scores.push_back(1.0f);  // detector responses are not comparable with the network scores
    }
    return features;
}

}  // namespace msckf
