// Bindings of the C++ XFeat front-end (module xfeat_cpp), used to compare it with the PyTorch implementation.
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "msckf_frontend/xfeat.hpp"

namespace py = pybind11;
using namespace msckf;

PYBIND11_MODULE(xfeat_cpp, m) {
    py::class_<XFeatExtractor>(m, "XFeatExtractor")
        .def(py::init([](const std::string& model_path, int top_k, bool use_coreml) {
                 XFeatExtractor::Options options;
                 options.top_k = top_k;
                 options.use_coreml = use_coreml;
                 return std::make_unique<XFeatExtractor>(model_path, options);
             }),
             py::arg("model_path"), py::arg("top_k") = 300, py::arg("use_coreml") = false)
        .def("extract", [](const XFeatExtractor& extractor, py::array_t<uint8_t, py::array::c_style | py::array::forcecast> image) {
            const cv::Mat grayscale(static_cast<int>(image.shape(0)), static_cast<int>(image.shape(1)), CV_8U, const_cast<uint8_t*>(image.data()));
            FrameFeatures features;
            {
                py::gil_scoped_release release;
                features = extractor.extract(grayscale);
            }
            Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> keypoints(features.keypoints.size(), 2);
            for (std::size_t k = 0; k < features.keypoints.size(); ++k) keypoints.row(static_cast<Eigen::Index>(k)) = features.keypoints[k].transpose();
            return py::make_tuple(keypoints, features.descriptors, features.scores);
        });
}
