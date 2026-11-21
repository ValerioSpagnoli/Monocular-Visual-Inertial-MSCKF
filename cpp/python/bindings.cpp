// Python bindings of the MSCKF core (module msckf_cpp), used to run the C++ filter from main.py and to compare it
// with the Python implementation.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "msckf/msckf.hpp"

namespace py = pybind11;
using namespace msckf;

namespace {

Pose makePose(const Mat3& R, const Vec3& t) { return Pose(R, t); }

std::vector<Vec2> rowsToPoints(const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>>& keypoints) {
    std::vector<Vec2> points(static_cast<std::size_t>(keypoints.rows()));
    for (Eigen::Index i = 0; i < keypoints.rows(); ++i) points[static_cast<std::size_t>(i)] = keypoints.row(i).transpose();
    return points;
}

}  // namespace

PYBIND11_MODULE(msckf_cpp, m) {
    m.doc() = "C++ core of the monocular visual-inertial MSCKF";

    py::class_<Parameters>(m, "Parameters")
        .def(py::init<>())
        .def_readwrite("K", &Parameters::K)
        .def_property("T_I_C_R", [](const Parameters& p) { return p.T_I_C.R; }, [](Parameters& p, const Mat3& R) { p.T_I_C.R = R; })
        .def_property("T_I_C_t", [](const Parameters& p) { return p.T_I_C.t; }, [](Parameters& p, const Vec3& t) { p.T_I_C.t = t; })
        .def_readwrite("sigma_image", &Parameters::sigma_image)
        .def_readwrite("only_imu", &Parameters::only_imu)
        .def_readwrite("accelerometer_noise_density", &Parameters::accelerometer_noise_density)
        .def_readwrite("accelerometer_random_walk", &Parameters::accelerometer_random_walk)
        .def_readwrite("gyroscope_noise_density", &Parameters::gyroscope_noise_density)
        .def_readwrite("gyroscope_random_walk", &Parameters::gyroscope_random_walk)
        .def_readwrite("gravity", &Parameters::gravity)
        .def_readwrite("estimate_initial_gyroscope_bias", &Parameters::estimate_initial_gyroscope_bias)
        .def_readwrite("initial_covariance_diagonal", &Parameters::initial_covariance_diagonal)
        .def_readwrite("min_cosine_similarity", &Parameters::min_cosine_similarity)
        .def_readwrite("use_parallax", &Parameters::use_parallax)
        .def_readwrite("min_parallax", &Parameters::min_parallax)
        .def_readwrite("epipolar_rejection_threshold", &Parameters::epipolar_rejection_threshold)
        .def_readwrite("homography_rejection_threshold", &Parameters::homography_rejection_threshold)
        .def_readwrite("min_number_of_frames_to_be_lost", &Parameters::min_number_of_frames_to_be_lost)
        .def_readwrite("min_number_of_frames_to_be_tracked", &Parameters::min_number_of_frames_to_be_tracked)
        .def_readwrite("max_number_of_camera_states", &Parameters::max_number_of_camera_states)
        .def_readwrite("max_triangulation_error", &Parameters::max_triangulation_error)
        .def_readwrite("zupt_max_disparity", &Parameters::zupt_max_disparity)
        .def_readwrite("zupt_sigma_velocity", &Parameters::zupt_sigma_velocity)
        .def_readwrite("max_track_length", &Parameters::max_track_length)
        .def_readwrite("update_features_on_prune", &Parameters::update_features_on_prune);

    py::class_<Msckf>(m, "Msckf")
        .def(py::init<const Parameters&>())
        .def("add_imu", [](Msckf& f, double timestamp, const Vec3& angular_velocity, const Vec3& linear_acceleration) {
            f.addImu({timestamp, angular_velocity, linear_acceleration});
        })
        .def("add_image", [](Msckf& f, const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>>& keypoints,
                             const Eigen::Ref<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& descriptors,
                             const std::vector<float>& scores) {
            FrameFeatures frame{rowsToPoints(keypoints), descriptors, scores};
            py::gil_scoped_release release;
            f.addImage(frame);
        })
        // IMU state
        .def_property_readonly("imu_id", [](const Msckf& f) { return f.imu().id; })
        .def_property_readonly("imu_initialized", [](const Msckf& f) { return f.imu().initialized; })
        .def_property_readonly("R_W_I", [](const Msckf& f) { return f.imu().T_W_I.R; })
        .def_property_readonly("p_W_I", [](const Msckf& f) { return f.imu().T_W_I.t; })
        .def_property_readonly("v_W_I", [](const Msckf& f) { return f.imu().v_W_I; })
        .def_property_readonly("gyroscope_bias", [](const Msckf& f) { return f.imu().gyroscope_bias; })
        .def_property_readonly("accelerometer_bias", [](const Msckf& f) { return f.imu().accelerometer_bias; })
        .def_property_readonly("covariance", [](const Msckf& f) { return f.covariance(); })
        .def_property_readonly("imu_covariance", [](const Msckf& f) { return MatX(f.covariance().topLeftCorner(15, 15)); })
        // Camera clones and features
        .def("cameras", [](const Msckf& f) {
            py::list out;
            for (const auto& [id, clone] : f.cameras()) out.append(py::make_tuple(id, clone.T_W_C.R, clone.T_W_C.t));
            return out;
        })
        .def("features", [](const Msckf& f) {
            py::dict out;
            for (const auto& [id, feature] : f.features()) {
                Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> keypoints(feature.keypoints.size(), 2);
                for (std::size_t k = 0; k < feature.keypoints.size(); ++k) keypoints.row(static_cast<Eigen::Index>(k)) = feature.keypoints[k].transpose();
                out[py::int_(id)] = py::make_tuple(keypoints, feature.camera_ids);
            }
            return out;
        })
        .def_property_readonly("number_of_features", [](const Msckf& f) { return f.features().size(); })
        // Diagnostics
        .def("rejected_matches", [](const Msckf& f) {
            py::list out;
            for (const auto& r : f.rejectedMatches()) out.append(py::make_tuple(r.keypoint, r.reason));
            return out;
        })
        .def_property_readonly("updated_feature_ids", &Msckf::updatedFeatureIds)
        .def_property_readonly("gated_feature_ids", &Msckf::gatedFeatureIds)
        .def("triangulated_points", [](const Msckf& f) { return f.triangulatedPoints(); })
        .def_property_readonly("epipolar_rejections", &Msckf::epipolarRejections)
        .def_property_readonly("homography_rejections", &Msckf::homographyRejections)
        .def_property_readonly("gating_rejections", &Msckf::gatingRejections)
        .def_property_readonly("triangulation_rejections", &Msckf::triangulationRejections)
        .def_property_readonly("zero_velocity_updates", &Msckf::zeroVelocityUpdates);

    // Building blocks, exposed for the parity tests against the Python implementation
    m.def("integrate", [](const Mat3& R, const Vec3& p, const Vec3& v, const Vec3& gravity, const Vec3& f, const Vec3& w, double dt) {
        ImuState imu;
        imu.T_W_I = makePose(R, p);
        imu.v_W_I = v;
        imu.gravity = gravity;
        imu.integrate(f, w, dt);
        return py::make_tuple(imu.T_W_I.R, imu.T_W_I.t, imu.v_W_I);
    });
    m.def("triangulate_feature", [](const std::vector<Mat3>& R, const std::vector<Vec3>& t, const std::vector<Vec2>& keypoints,
                                    const Mat3& invK, const Vec3& initial_point) {
        std::vector<Pose> poses;
        for (std::size_t i = 0; i < R.size(); ++i) poses.push_back(makePose(R[i], t[i]));
        const TriangulationResult result = triangulateFeature(poses, keypoints, invK, initial_point);
        return py::make_tuple(result.point ? py::cast(*result.point) : py::none(), result.rms);
    });
    m.def("sampson_distance", &sampsonDistance);
    m.def("chi2_quantile", &chi2Quantile, py::arg("dof"), py::arg("probability") = 0.95);
    m.def("observability_constrained_transition", [](const MatX& Phi, double dt, const Mat3& R_null, const Vec3& p_null, const Vec3& v_null,
                                                     const Mat3& R, const Vec3& p, const Vec3& v, const Vec3& gravity) {
        return observabilityConstrainedTransition(Phi, dt, makePose(R_null, p_null), v_null, makePose(R, p), v, gravity);
    });
}
