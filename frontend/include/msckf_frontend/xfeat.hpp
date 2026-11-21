#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "msckf/math.hpp"

namespace msckf {

/// Keypoints, descriptors and detection scores of one image.
struct FrameFeatures {
    std::vector<Vec2> keypoints;           // [px], in the undistorted image
    Eigen::MatrixXf descriptors;           // one L2-normalized row per keypoint
    std::vector<float> scores;
};

/// XFeat keypoints and descriptors (Potje et al., CVPR 2024) with ONNX Runtime. The network (models/xfeat.onnx,
/// exported by tools/export_xfeat_onnx.py) returns the dense descriptors, the keypoint heatmap and the reliability
/// map; NMS, scores, top-k selection and descriptor sampling reproduce XFeat.detectAndCompute.
class XFeatExtractor {
public:
    struct Options {
        int top_k = 300;
        float detection_threshold = 0.05f;
        int nms_kernel = 5;
        int threads = 0;          // 0: ONNX Runtime default
        bool use_coreml = false;  // Apple GPU / Neural Engine (macOS, iOS)
    };

    XFeatExtractor(const std::string& model_path, const Options& options);
    ~XFeatExtractor();

    /// Keypoints [px] in the input image, L2-normalized descriptors and scores, sorted by decreasing score.
    FrameFeatures extract(const cv::Mat& grayscale) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Options options_;
};

}  // namespace msckf
