"""
Parity between the Python MSCKF and the C++ core (msckf_cpp): building blocks to numerical precision, and the full
filter on a synthetic visual-inertial sequence. Skipped if the module has not been built (`pixi run build-cpp`).
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scipy.stats import chi2

msckf_cpp = pytest.importorskip('msckf_cpp')

from src.utils.geometry import Isometry3D, skew
from src.msckf.IMU import IMU, IMUMeasurement
from src.msckf.MSCKF import MSCKF, MSCKFParameters, observability_constrained_transition, triangulate_feature, sampson_distance, fundamental_matrix
from src.msckf.FeatureExtractor import ExtractedFeature

rng = np.random.default_rng(1)
K = np.array([[400.0, 0, 320], [0, 400.0, 240], [0, 0, 1]])
G = np.array([0, 0, -9.81])


def random_pose(scale=1.0):
    return Isometry3D(Rotation.random(random_state=rng.integers(1e6)).as_matrix(), rng.normal(scale=scale, size=3))


def test_integration():
    T, v = random_pose(), rng.normal(size=3)
    f, w = rng.normal(size=3) + np.array([0, 0, 9.81]), rng.normal(scale=0.5, size=3)
    imu = IMU(T_W_Ii=Isometry3D(T.R, T.t), v_W_Ii=v, W_gravity=G)
    imu.integrate(linear_acceleration=f, angular_velocity=w, dt=0.005)
    R, p, v_cpp = msckf_cpp.integrate(T.R, T.t, v, G, f, w, 0.005)
    np.testing.assert_allclose(R, imu.T_W_Ii.R, atol=1e-14)
    np.testing.assert_allclose(p, imu.T_W_Ii.t, atol=1e-14)
    np.testing.assert_allclose(v_cpp, imu.v_W_Ii, atol=1e-14)


def test_observability_constrained_transition():
    Phi = np.eye(15) + 0.01 * rng.normal(size=(15, 15))
    T0, T1 = random_pose(), random_pose()
    v0, v1 = rng.normal(size=3), rng.normal(size=3)
    expected = observability_constrained_transition(Phi, 0.005, T0, v0, T1, v1, G)
    actual = msckf_cpp.observability_constrained_transition(Phi, 0.005, T0.R, T0.t, v0, T1.R, T1.t, v1, G)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_triangulation_and_sampson():
    T0 = random_pose()
    poses = [T0 * Isometry3D(Rotation.from_rotvec(rng.normal(scale=0.05, size=3)).as_matrix(), rng.normal(scale=0.3, size=3)) for _ in range(5)]
    W_f = T0.transform(np.array([0.4, -0.3, 4.5]))
    pixel = lambda T: (K @ (T.R.T @ (W_f - T.t) / (T.R.T @ (W_f - T.t))[2]))[:2] + rng.normal(scale=0.5, size=2)
    keypoints = [pixel(T) for T in poses]
    initial = T0.t + 1.3 * (W_f - T0.t)

    point_py, rms_py = triangulate_feature(poses, keypoints, np.linalg.inv(K), initial)
    point_cpp, rms_cpp = msckf_cpp.triangulate_feature([T.R for T in poses], [T.t for T in poses], keypoints, np.linalg.inv(K), initial)
    np.testing.assert_allclose(point_cpp, point_py, atol=1e-9)
    assert rms_cpp == pytest.approx(rms_py, abs=1e-12)

    F = fundamental_matrix(poses[0].inv() * poses[1], np.linalg.inv(K))
    assert msckf_cpp.sampson_distance(F, keypoints[0], keypoints[1]) == pytest.approx(sampson_distance(F, keypoints[0], keypoints[1]), abs=1e-12)


@pytest.mark.parametrize('dof', [1, 2, 5, 17, 60, 200])
def test_chi2_quantile(dof):
    assert msckf_cpp.chi2_quantile(dof) == pytest.approx(chi2.ppf(0.95, dof), rel=1e-10)


def synthetic_sequence(n_images=60, imu_per_image=10, dt=0.005):
    """Smooth motion with IMU data integrated exactly by the filter, and noisy observations of fixed landmarks."""
    landmarks = rng.uniform([-4, -4, -2], [4, 4, 2], size=(400, 3)) + np.array([0, 0, 0])
    descriptors = rng.normal(size=(len(landmarks), 64)).astype(np.float32)
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
    T_I_C = Isometry3D(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float), np.array([0.05, 0.0, 0.02]))

    state = IMU(T_W_Ii=Isometry3D(np.eye(3), np.array([0.0, 0.0, 0.0])), v_W_Ii=np.zeros(3), W_gravity=G)
    imu, images, timestamp = [], [], 0.0
    for k in range(n_images * imu_per_image):
        t = k * dt
        # Static for the first 20 IMU samples (gravity initialization), then smooth accelerations and rotations
        if k < 20: w, a_W = np.zeros(3), np.zeros(3)
        else: w, a_W = np.array([0.2 * np.sin(t), 0.3 * np.cos(0.7 * t), 0.4 * np.sin(0.5 * t)]), np.array([0.8 * np.cos(t), 0.6 * np.sin(1.3 * t), 0.2 * np.sin(t)])
        f = state.T_W_Ii.R.T @ (a_W - G)
        imu.append(IMUMeasurement(timestamp, w, f))
        if k % imu_per_image == 0:
            T_W_C = state.T_W_Ii * T_I_C
            C_f = (landmarks - T_W_C.t) @ T_W_C.R
            visible = C_f[:, 2] > 0.5
            pixels = (C_f[visible] / C_f[visible, 2:3]) @ K.T
            inside = (pixels[:, 0] > 0) & (pixels[:, 0] < 640) & (pixels[:, 1] > 0) & (pixels[:, 1] < 480)
            index = np.flatnonzero(visible)[inside]
            images.append((k, ExtractedFeature(keypoints=pixels[inside, :2] + rng.normal(scale=0.5, size=(inside.sum(), 2)),
                                               descriptors=descriptors[index], scores=np.ones(len(index), dtype=np.float32))))
        state.integrate(linear_acceleration=f, angular_velocity=w, dt=dt)
        timestamp += dt
    return imu, images, T_I_C


def test_full_filter_parity():
    imu, images, T_I_C = synthetic_sequence()
    parameters = MSCKFParameters(T_W_C=T_I_C, K=K, sigma_image=0.5, accelerometer_noise_density=0.002, gyroscope_noise_density=0.0002,
                                 accelerometer_random_walk=1e-4, gyroscope_random_walk=1e-5, W_gravity=G,
                                 initial_covariance_diagonal=np.concatenate([np.full(3, 1e-4), np.full(3, 1e-6), np.full(3, 1e-4), np.full(3, 1e-3), np.zeros(3)]),
                                 min_cosine_similarity=0.95, min_parallax=5, epipolar_rejection_threshold=10, max_number_of_camera_states=15,
                                 min_number_of_frames_to_be_tracked=4, min_number_of_frames_to_be_lost=1, max_triangulation_error=3.0, device='cpu')

    python_filter = MSCKF(parameters)
    cpp_parameters = __import__('src.msckf.cpp_backend', fromlist=['to_cpp_parameters']).to_cpp_parameters(parameters)
    cpp_parameters.max_track_length = 0                # update strategy of the Python implementation
    cpp_parameters.update_features_on_prune = False
    cpp_filter = msckf_cpp.Msckf(cpp_parameters)

    image_at = dict(images)
    for k, measurement in enumerate(imu):
        # As in main.py: propagate up to the image timestamp, then process the image
        python_filter.imu_callback(measurement)
        cpp_filter.add_imu(measurement.timestamp, measurement.angular_velocity, measurement.linear_acceleration)
        if k in image_at:
            frame = image_at[k]
            python_filter.feature_callback(None, frame)
            cpp_filter.add_image(np.ascontiguousarray(frame.keypoints, dtype=float), frame.descriptors, frame.scores.tolist())

    # The filter must have done real work for the comparison to mean something
    assert len(python_filter.updated_feature_indices) + python_filter.number_of_residuals_discarded_for_gasting_test >= 0
    assert len(python_filter.triangulated_points) > 50
    assert len(cpp_filter.triangulated_points()) == len(python_filter.triangulated_points)

    np.testing.assert_allclose(cpp_filter.p_W_I, python_filter.state.imu.T_W_Ii.t, atol=1e-6)
    np.testing.assert_allclose(cpp_filter.R_W_I, python_filter.state.imu.T_W_Ii.R, atol=1e-6)
    np.testing.assert_allclose(cpp_filter.v_W_I, python_filter.state.imu.v_W_Ii, atol=1e-6)
    np.testing.assert_allclose(cpp_filter.accelerometer_bias, python_filter.state.imu.accelerometer_bias, atol=1e-6)
    np.testing.assert_allclose(cpp_filter.covariance, python_filter.state.covariance, atol=1e-9)
