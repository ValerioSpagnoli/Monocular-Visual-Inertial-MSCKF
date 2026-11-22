#pragma once

#include <memory>
#include <string>

#include "msckf/feature_extractor.hpp"

namespace msckf {

/// XFeat keypoints and descriptors (Potje et al., CVPR 2024) with ONNX Runtime. The network (models/xfeat.onnx,
/// exported by tools/export_xfeat_onnx.py) returns the dense descriptors, the keypoint heatmap and the reliability
/// map; NMS, scores, top-k selection and descriptor sampling reproduce XFeat.detectAndCompute.
/// LiftFeat (Liu et al., ICRA 2025), built on XFeat, is exported with the same outputs (tools/export_liftfeat_onnx.py,
/// constant reliability) and runs with this class as well, with detection_threshold 0.1.
class XFeatExtractor : public FeatureExtractor {
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
    FrameFeatures extract(const cv::Mat& grayscale) const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Options options_;
};

}  // namespace msckf
