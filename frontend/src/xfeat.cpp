#include "msckf_frontend/xfeat.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

#include <onnxruntime_cxx_api.h>
#if __has_include(<coreml_provider_factory.h>)
#include <coreml_provider_factory.h>
#define MSCKF_HAS_COREML 1
#endif
#include <opencv2/imgproc.hpp>

namespace msckf {

namespace {

/// Sampling of a (channels, height, width) map at the keypoint x, y of the network input (size W x H), as
/// torch.nn.functional.grid_sample with align_corners=False and zero padding after XFeat's normalization
/// 2 x / (W - 1) - 1: the sample position in the map is x * width / (W - 1) - 0.5.
struct MapView {
    const float* data;
    int channels, height, width;
    float at(int c, int y, int x) const {
        return (x < 0 || y < 0 || x >= width || y >= height) ? 0.0f : data[(static_cast<std::size_t>(c) * height + y) * width + x];
    }
};

std::pair<double, double> samplePosition(double x, double y, int W, int H, const MapView& map) {
    return {x * map.width / (W - 1.0) - 0.5, y * map.height / (H - 1.0) - 0.5};
}

float sampleNearest(const MapView& map, double ix, double iy) {
    return map.at(0, static_cast<int>(std::nearbyint(iy)), static_cast<int>(std::nearbyint(ix)));  // round half to even
}

float sampleBilinear(const MapView& map, double ix, double iy) {
    const int x0 = static_cast<int>(std::floor(ix)), y0 = static_cast<int>(std::floor(iy));
    const double tx = ix - x0, ty = iy - y0;
    return static_cast<float>((1 - tx) * (1 - ty) * map.at(0, y0, x0) + tx * (1 - ty) * map.at(0, y0, x0 + 1) +
                              (1 - tx) * ty * map.at(0, y0 + 1, x0) + tx * ty * map.at(0, y0 + 1, x0 + 1));
}

/// Cubic convolution weights of PyTorch (A = -0.75) for the taps at -1, 0, 1, 2 around floor(position).
std::array<double, 4> cubicWeights(double t) {
    constexpr double A = -0.75;
    auto near = [](double x) { return ((A + 2) * x - (A + 3)) * x * x + 1; };
    auto far = [](double x) { return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A; };
    return {far(t + 1), near(t), near(1 - t), far(2 - t)};
}

void sampleBicubic(const MapView& map, double ix, double iy, float* out) {
    const int x0 = static_cast<int>(std::floor(ix)), y0 = static_cast<int>(std::floor(iy));
    const std::array<double, 4> wx = cubicWeights(ix - x0), wy = cubicWeights(iy - y0);
    for (int c = 0; c < map.channels; ++c) {
        double value = 0.0;
        for (int j = 0; j < 4; ++j) {
            double row = 0.0;
            for (int i = 0; i < 4; ++i) row += wx[i] * map.at(c, y0 - 1 + j, x0 - 1 + i);
            value += wy[j] * row;
        }
        out[c] = static_cast<float>(value);
    }
}

}  // namespace

struct XFeatExtractor::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "xfeat"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
};

XFeatExtractor::XFeatExtractor(const std::string& model_path, const Options& options) : impl_(std::make_unique<Impl>()), options_(options) {
    if (options.threads > 0) impl_->session_options.SetIntraOpNumThreads(options.threads);
    impl_->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef MSCKF_HAS_COREML
    if (options.use_coreml) Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(impl_->session_options, 0));
#endif
    impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), impl_->session_options);
}

XFeatExtractor::~XFeatExtractor() = default;

FrameFeatures XFeatExtractor::extract(const cv::Mat& grayscale) const {
    // Network input: size divisible by 32 (bilinear resize, as XFeat.preprocess_tensor)
    const int W = (grayscale.cols / 32) * 32, H = (grayscale.rows / 32) * 32;
    const double rw = static_cast<double>(grayscale.cols) / W, rh = static_cast<double>(grayscale.rows) / H;
    cv::Mat input;
    grayscale.convertTo(input, CV_32F);
    if (input.cols != W || input.rows != H) cv::resize(input, input, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
    if (!input.isContinuous()) input = input.clone();

    const std::array<int64_t, 4> shape{1, 1, H, W};
    const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value image = Ort::Value::CreateTensor<float>(memory, input.ptr<float>(), static_cast<std::size_t>(W) * H, shape.data(), shape.size());
    const char* input_names[] = {"image"};
    const char* output_names[] = {"descriptors", "heatmap", "reliability"};
    std::vector<Ort::Value> outputs = impl_->session->Run(Ort::RunOptions{nullptr}, input_names, &image, 1, output_names, 3);

    auto view = [](const Ort::Value& value) {
        const auto dims = value.GetTensorTypeAndShapeInfo().GetShape();
        return MapView{value.GetTensorData<float>(), static_cast<int>(dims[1]), static_cast<int>(dims[2]), static_cast<int>(dims[3])};
    };
    const MapView descriptors = view(outputs[0]), heatmap = view(outputs[1]), reliability = view(outputs[2]);

    // NMS: local maxima of the heatmap in a kernel x kernel window above the threshold
    const cv::Mat heat(H, W, CV_32F, const_cast<float*>(heatmap.data));
    cv::Mat local_max;
    cv::dilate(heat, local_max, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(options_.nms_kernel, options_.nms_kernel)));
    std::vector<std::array<int, 2>> candidates;  // x, y
    for (int y = 0; y < H; ++y) {
        const float* h = heat.ptr<float>(y);
        const float* m = local_max.ptr<float>(y);
        for (int x = 0; x < W; ++x)
            if (h[x] == m[x] && h[x] > options_.detection_threshold) candidates.push_back({x, y});
    }

    // Scores: heatmap (nearest) times reliability (bilinear), then the top-k
    std::vector<float> scores(candidates.size());
    for (std::size_t k = 0; k < candidates.size(); ++k) {
        const auto [x, y] = candidates[k];
        if (x == 0 && y == 0) { scores[k] = -1.0f; continue; }
        const auto [hx, hy] = samplePosition(x, y, W, H, heatmap);
        const auto [rx, ry] = samplePosition(x, y, W, H, reliability);
        scores[k] = sampleNearest(heatmap, hx, hy) * sampleBilinear(reliability, rx, ry);
    }
    std::vector<std::size_t> order(candidates.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return scores[a] > scores[b]; });
    if (static_cast<int>(order.size()) > options_.top_k) order.resize(options_.top_k);

    // Descriptors: bicubic sampling of the dense map, L2-normalized
    FrameFeatures features;
    features.descriptors.resize(static_cast<Eigen::Index>(order.size()), descriptors.channels);
    Eigen::Index row = 0;
    std::vector<float> descriptor(descriptors.channels);
    for (std::size_t k : order) {
        if (scores[k] <= 0.0f) continue;
        const auto [x, y] = candidates[k];
        const auto [dx, dy] = samplePosition(x, y, W, H, descriptors);
        sampleBicubic(descriptors, dx, dy, descriptor.data());
        const Eigen::Map<Eigen::VectorXf> d(descriptor.data(), descriptors.channels);
        features.descriptors.row(row++) = d.transpose() / std::max(d.norm(), 1e-12f);
        features.keypoints.emplace_back(x * rw, y * rh);
        features.scores.push_back(scores[k]);
    }
    features.descriptors.conservativeResize(row, Eigen::NoChange);
    return features;
}

}  // namespace msckf
