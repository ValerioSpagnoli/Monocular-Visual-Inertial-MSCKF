"""
Adapter running the C++ MSCKF core (module msckf_cpp, built from cpp/) behind the interface of the Python MSCKF class,
so that main.py, the Rerun visualizer and the metrics work unchanged with either backend.

Build the module first with `pixi run build-cpp`.
"""
from types import SimpleNamespace
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.msckf.FeatureExtractor import FeatureExtractor, ExtractedFeature
from src.msckf.IMU import IMUMeasurement
from src.utils.geometry import Isometry3D

try:
    import msckf_cpp
except ImportError as error:
    raise ImportError('msckf_cpp not found: build it with `pixi run build-cpp`') from error


def to_cpp_parameters(parameters) -> 'msckf_cpp.Parameters':
    p = msckf_cpp.Parameters()
    p.K = np.asarray(parameters.K, dtype=float)
    # Camera pose in the IMU frame (the IMU frame of the Python filter is the identity at start, so T_I_C = T_W_C)
    p.T_I_C_R = np.asarray(parameters.T_W_C.R, dtype=float)
    p.T_I_C_t = np.asarray(parameters.T_W_C.t, dtype=float)
    p.sigma_image = parameters.sigma_image
    p.only_imu = parameters.only_imu
    p.accelerometer_noise_density = parameters.accelerometer_noise_density
    p.accelerometer_random_walk = parameters.accelerometer_random_walk
    p.gyroscope_noise_density = parameters.gyroscope_noise_density
    p.gyroscope_random_walk = parameters.gyroscope_random_walk
    p.gravity = np.asarray(parameters.W_gravity, dtype=float)
    p.estimate_initial_gyroscope_bias = parameters.estimate_initial_gyroscope_bias
    p.initial_covariance_diagonal = np.asarray(parameters.initial_covariance_diagonal, dtype=float)
    p.min_cosine_similarity = parameters.min_cosine_similarity
    p.use_parallax = parameters.use_parallax
    p.min_parallax = parameters.min_parallax
    p.epipolar_rejection_threshold = parameters.epipolar_rejection_threshold
    p.homography_rejection_threshold = parameters.homography_rejection_threshold
    p.min_number_of_frames_to_be_lost = parameters.min_number_of_frames_to_be_lost
    p.min_number_of_frames_to_be_tracked = parameters.min_number_of_frames_to_be_tracked
    p.max_number_of_camera_states = parameters.max_number_of_camera_states
    p.max_triangulation_error = parameters.max_triangulation_error
    p.zupt_max_disparity = parameters.zupt_max_disparity
    p.zupt_sigma_velocity = parameters.zupt_sigma_velocity
    return p


class _ImuView:
    def __init__(self, core):
        self._core = core

    @property
    def T_W_Ii(self) -> Isometry3D: return Isometry3D(self._core.R_W_I, self._core.p_W_I)
    @property
    def v_W_Ii(self) -> np.ndarray: return self._core.v_W_I
    @property
    def gyroscope_bias(self) -> np.ndarray: return self._core.gyroscope_bias
    @property
    def accelerometer_bias(self) -> np.ndarray: return self._core.accelerometer_bias
    @property
    def id(self) -> int: return self._core.imu_id
    @property
    def is_initialized(self) -> bool: return self._core.imu_initialized


class _StateView:
    def __init__(self, core):
        self._core = core
        self.imu = _ImuView(core)

    @property
    def covariance(self) -> np.ndarray: return self._core.covariance

    @property
    def cameras(self) -> Dict[int, SimpleNamespace]:
        return {camera_id: SimpleNamespace(T_W_Ci=Isometry3D(R, t)) for camera_id, R, t in self._core.cameras()}


class CppMSCKF:
    def __init__(self, parameters, rr=None):
        self.core = msckf_cpp.Msckf(to_cpp_parameters(parameters))
        self.state = _StateView(self.core)
        self.feature_extractor = FeatureExtractor(device=parameters.device)
        self.number_of_extracted_features = parameters.number_of_extracted_features

    def imu_callback(self, imu_measurement: IMUMeasurement):
        self.core.add_imu(float(imu_measurement.timestamp), np.asarray(imu_measurement.angular_velocity, dtype=float),
                          np.asarray(imu_measurement.linear_acceleration, dtype=float))

    def feature_callback(self, image: np.ndarray, extracted_features: ExtractedFeature = None):
        if extracted_features is None:
            extracted_features = self.feature_extractor.extract_features(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), self.number_of_extracted_features)
        self.core.add_image(np.ascontiguousarray(extracted_features.keypoints, dtype=np.float64),
                            np.ascontiguousarray(extracted_features.descriptors, dtype=np.float32),
                            np.asarray(extracted_features.scores, dtype=np.float32).tolist())

    #* Views used by main.py and the visualizer
    @property
    def features(self) -> Dict[int, SimpleNamespace]:
        return {feature_id: SimpleNamespace(keypoints=list(keypoints), camera_indices=list(camera_ids))
                for feature_id, (keypoints, camera_ids) in self.core.features().items()}

    @property
    def rejected_keypoints(self) -> List[Tuple[np.ndarray, str]]: return self.core.rejected_matches()
    @property
    def updated_feature_indices(self) -> List[int]: return self.core.updated_feature_ids
    @property
    def gated_feature_indices(self) -> List[int]: return self.core.gated_feature_ids
    @property
    def triangulated_points(self) -> Dict[int, np.ndarray]: return self.core.triangulated_points()

    @property
    def number_of_features_discarded_for_epipolar_test(self) -> int: return self.core.epipolar_rejections
    @property
    def number_of_features_discarder_for_homography_test(self) -> int: return self.core.homography_rejections
    @property
    def number_of_residuals_discarded_for_gasting_test(self) -> int: return self.core.gating_rejections
    @property
    def number_of_features_discarded_for_triangulation(self) -> int: return self.core.triangulation_rejections
    @property
    def number_of_zero_velocity_updates(self) -> int: return self.core.zero_velocity_updates
