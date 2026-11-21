"""
Numerical checks of the MSCKF Jacobians against the error-state convention of the filter
(local orientation error, as in the JPL formulation of Mourikis and Hesch):

    R_true = R_est @ exp(δθ)        p_true = p_est + δp

with R the rotation from the body (IMU or camera) frame to the world frame.

A linearized model H is correct when h(x_est ⊞ δx) - h(x_est) ≈ H δx for a small δx.
"""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.utils.geometry import Isometry3D, skew
from src.msckf.Camera import Camera
from src.msckf.IMU import IMU
from src.msckf.MSCKF import MSCKF, MSCKFState, observability_constrained_transition, epipolar_residual, triangulate_feature

rng = np.random.default_rng(0)
EPS = 1e-6
K = np.array([[400.0, 0, 320], [0, 400.0, 240], [0, 0, 1]])


def exp_so3(phi: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(phi).as_matrix()

def log_so3(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_rotvec()

def apply_correction(T: Isometry3D, dtheta: np.ndarray, dp: np.ndarray) -> Isometry3D:
    return Isometry3D(T.R @ exp_so3(dtheta), T.t + dp)

def orientation_error(R_true: np.ndarray, R_est: np.ndarray) -> np.ndarray:
    """Inverse of apply_correction: δθ such that R_true = R_est @ exp(δθ)"""
    return log_so3(R_est.T @ R_true)

def random_pose() -> Isometry3D:
    return Isometry3D(Rotation.random(random_state=rng.integers(1e6)).as_matrix(), rng.normal(size=3))

def point_in_front(T_W_C: Isometry3D) -> np.ndarray:
    Ci_f = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(3, 6)])
    return T_W_C.transform(Ci_f)

def project(T_W_C: Isometry3D, W_f: np.ndarray) -> np.ndarray:
    Ci_f = T_W_C.R.T @ (W_f - T_W_C.t)
    return Ci_f[:2] / Ci_f[2]

def numerical_jacobian(f, n: int) -> np.ndarray:
    columns = []
    for k in range(n):
        d = np.zeros(n); d[k] = EPS
        columns.append((f(d) - f(-d)) / (2 * EPS))
    return np.stack(columns, axis=1)


@pytest.mark.parametrize('trial', range(5))
def test_camera_measurement_jacobian_wrt_camera_pose(trial):
    T_W_C = random_pose()
    W_f = point_in_front(T_W_C)
    camera = Camera(K=K, width=640, height=480, T_W_Ci=T_W_C)
    H_x, _ = camera.compute_jacobians(T_W_C.R.T @ (W_f - T_W_C.t))

    numerical = numerical_jacobian(lambda d: project(apply_correction(T_W_C, d[:3], d[3:]), W_f), 6)
    np.testing.assert_allclose(H_x[:, 3:], numerical[:, 3:], atol=1e-6, err_msg='position block')
    np.testing.assert_allclose(H_x[:, :3], numerical[:, :3], atol=1e-6, err_msg='orientation block')


@pytest.mark.parametrize('trial', range(5))
def test_camera_measurement_jacobian_wrt_feature(trial):
    T_W_C = random_pose()
    W_f = point_in_front(T_W_C)
    camera = Camera(K=K, width=640, height=480, T_W_Ci=T_W_C)
    H_x, H_f = camera.compute_jacobians(T_W_C.R.T @ (W_f - T_W_C.t))

    numerical = numerical_jacobian(lambda d: project(T_W_C, W_f + d), 3)
    np.testing.assert_allclose(H_f, numerical, atol=1e-6)
    # MSCKF.compute_residual_and_jacobians uses H_f = -H_x[:, 3:]
    np.testing.assert_allclose(-H_x[:, 3:], numerical, atol=1e-6)


def fake_msckf(T_W_I: Isometry3D, T_I_C: Isometry3D, covariance: np.ndarray, cameras: dict = None) -> SimpleNamespace:
    """Minimal object exposing the attributes used by the MSCKF methods under test (avoids loading XFeat)."""
    imu = IMU(T_W_I=Isometry3D(np.eye(3), np.zeros(3)), T_W_Ii=T_W_I)
    return SimpleNamespace(state=MSCKFState(imu=imu, cameras=cameras or {}, covariance=covariance),
                           T_W_C=T_I_C, K=K, width=640, height=480)


@pytest.mark.parametrize('trial', range(5))
def test_state_augmentation_jacobian(trial):
    """Camera clone error [δθ_C, δp_C] as a function of the IMU error (MSCKF.state_augmentation)."""
    T_W_I = random_pose()
    T_I_C = Isometry3D(Rotation.random(random_state=rng.integers(1e6)).as_matrix(), rng.normal(scale=0.1, size=3))
    T_W_C = T_W_I * T_I_C
    
    # With an identity covariance the cross-covariance block of the new clone is the augmentation Jacobian J
    msckf = fake_msckf(T_W_I, T_I_C, np.eye(15))
    MSCKF.state_augmentation(msckf)
    J = msckf.state.covariance[15:21, :15]
    J = np.hstack([J[:, 0:3], J[:, 12:15]]) # columns of [δθ_I, δp_I]
    
    def camera_error(d):
        T_W_C_perturbed = apply_correction(T_W_I, d[:3], d[3:]) * T_I_C
        return np.concatenate([orientation_error(T_W_C_perturbed.R, T_W_C.R), T_W_C_perturbed.t - T_W_C.t])
    
    numerical = numerical_jacobian(camera_error, 6)
    np.testing.assert_allclose(J[:3], numerical[:3], atol=1e-6, err_msg='orientation rows')
    np.testing.assert_allclose(J[3:], numerical[3:], atol=1e-6, err_msg='position rows')


def test_correction_follows_error_convention():
    """MSCKF.correct() must apply δx with the same convention used to derive the Jacobians."""
    T_W_I, T_W_C = random_pose(), random_pose()
    camera = Camera(K=K, width=640, height=480, T_W_Ci=Isometry3D(T_W_C.R.copy(), T_W_C.t.copy()))
    msckf = fake_msckf(Isometry3D(T_W_I.R.copy(), T_W_I.t.copy()), T_W_C, np.eye(21), cameras={0: camera})
    
    delta_x = rng.normal(scale=0.01, size=(21, 1))
    MSCKF.correct(msckf, np.zeros((21, 1)), np.zeros((1, 21)), np.zeros((1, 1)), delta_x)
    
    expected_imu = apply_correction(T_W_I, delta_x[0:3, 0], delta_x[12:15, 0])
    expected_camera = apply_correction(T_W_C, delta_x[15:18, 0], delta_x[18:21, 0])
    np.testing.assert_allclose(msckf.state.imu.T_W_Ii.R, expected_imu.R, atol=1e-9, err_msg='IMU orientation')
    np.testing.assert_allclose(msckf.state.imu.T_W_Ii.t, expected_imu.t, atol=1e-9, err_msg='IMU position')
    np.testing.assert_allclose(camera.T_W_Ci.R, expected_camera.R, atol=1e-9, err_msg='camera orientation')
    np.testing.assert_allclose(camera.T_W_Ci.t, expected_camera.t, atol=1e-9, err_msg='camera position')


def test_imu_propagation_jacobian():
    """Discrete error transition Φ ≈ I + F dt (MSCKF.process_imu) against two integrated trajectories."""
    dt = 0.005
    omega = np.array([0.3, -0.2, 0.5])
    acc = np.array([0.4, 0.1, -9.6])
    W_gravity = np.array([0, 0, -9.81])

    def propagate(T_W_I, v):
        imu = IMU(T_W_Ii=T_W_I, v_W_Ii=v.copy(), W_gravity=W_gravity)
        imu.integrate(linear_acceleration=acc, angular_velocity=omega, dt=dt)
        return imu.T_W_Ii, imu.v_W_Ii

    T0, v0 = random_pose(), rng.normal(size=3)
    T1, v1 = propagate(T0, v0)

    # F as in MSCKF.process_imu (planet rotation neglected), on the [δθ, δv, δp] blocks
    F = np.zeros((9, 9))
    F[0:3, 0:3] = -skew(omega)
    F[3:6, 0:3] = -T0.R @ skew(acc)
    F[6:9, 3:6] = np.eye(3)
    Phi = np.eye(9) + F * dt

    def error_after_propagation(d):
        T1p, v1p = propagate(apply_correction(T0, d[:3], d[6:9]), v0 + d[3:6])
        return np.concatenate([orientation_error(T1p.R, T1.R), v1p - v1, T1p.t - T1.t])

    numerical = numerical_jacobian(error_after_propagation, 9)
    np.testing.assert_allclose(Phi[0:3, 0:3], numerical[0:3, 0:3], atol=1e-4, err_msg='δθ <- δθ')
    np.testing.assert_allclose(Phi[3:6, 0:3], numerical[3:6, 0:3], atol=1e-4, err_msg='δv <- δθ')


def test_observability_constrained_transition():
    """With the null state equal to the state before propagation, the OC transition must still match the true error dynamics."""
    dt = 0.005
    omega = np.array([0.3, -0.2, 0.5])
    acc = np.array([0.4, 0.1, -9.6])
    W_gravity = np.array([0, 0, -9.81])
    
    def propagate(T_W_I, v):
        imu = IMU(T_W_Ii=T_W_I, v_W_Ii=v.copy(), W_gravity=W_gravity)
        imu.integrate(linear_acceleration=acc, angular_velocity=omega, dt=dt)
        return imu.T_W_Ii, imu.v_W_Ii
    
    T0, v0 = random_pose(), rng.normal(size=3)
    T1, v1 = propagate(T0, v0)
    
    # Unconstrained Φ ≈ I + F dt on the full 15-dim state, as in MSCKF.process_imu
    F = np.zeros((15, 15))
    F[0:3, 0:3] = -skew(omega)
    F[0:3, 3:6] = -np.eye(3)
    F[6:9, 0:3] = -T0.R @ skew(acc)
    F[6:9, 9:12] = -T0.R
    F[12:15, 6:9] = np.eye(3)
    Phi = observability_constrained_transition(np.eye(15) + F * dt, dt, T0, v0, T1, v1, W_gravity)
    
    def error_after_propagation(d):
        T1p, v1p = propagate(apply_correction(T0, d[:3], d[6:9]), v0 + d[3:6])
        return np.concatenate([orientation_error(T1p.R, T1.R), v1p - v1, T1p.t - T1.t])
    
    numerical = numerical_jacobian(error_after_propagation, 9)
    np.testing.assert_allclose(Phi[0:3, 0:3], numerical[0:3, 0:3], atol=1e-6, err_msg='δθ <- δθ')
    np.testing.assert_allclose(Phi[6:9, 0:3], numerical[3:6, 0:3], atol=1e-3, err_msg='δv <- δθ')
    np.testing.assert_allclose(Phi[12:15, 0:3], numerical[6:9, 0:3], atol=1e-4, err_msg='δp <- δθ')


def test_correction_keeps_null_states():
    """The first-estimate (null) states of the observability constraint must not be modified by MSCKF.correct()."""
    T_W_I, T_W_C = random_pose(), random_pose()
    camera = Camera(K=K, width=640, height=480, T_W_Ci=Isometry3D(T_W_C.R.copy(), T_W_C.t.copy()))
    msckf = fake_msckf(Isometry3D(T_W_I.R.copy(), T_W_I.t.copy()), T_W_C, np.eye(21), cameras={0: camera})
    imu = msckf.state.imu
    imu.v_W_Ii = rng.normal(size=3)
    imu.T_W_Ii_null, imu.v_W_Ii_null = imu.T_W_Ii, imu.v_W_Ii # as set at the end of MSCKF.process_imu
    
    MSCKF.correct(msckf, np.zeros((21, 1)), np.zeros((1, 21)), np.zeros((1, 1)), rng.normal(scale=0.01, size=(21, 1)))
    
    np.testing.assert_array_equal(imu.T_W_Ii_null.R, T_W_I.R)
    np.testing.assert_array_equal(imu.T_W_Ii_null.t, T_W_I.t)
    np.testing.assert_array_equal(camera.T_W_Ci_null.R, T_W_C.R)
    np.testing.assert_array_equal(camera.T_W_Ci_null.t, T_W_C.t)
    assert not np.allclose(imu.v_W_Ii_null, imu.v_W_Ii)


@pytest.mark.parametrize('trial', range(5))
def test_epipolar_residual(trial):
    """Zero for two images of the same 3D point, clearly positive for a wrong match."""
    T_W_C1 = random_pose()
    T_W_C2 = T_W_C1 * Isometry3D(exp_so3(rng.normal(scale=0.1, size=3)), rng.normal(scale=0.3, size=3))
    W_f = point_in_front(T_W_C1)
    pixel = lambda T, p: (K @ np.append(project(T, p), 1))[:2]
    keypoint_1, keypoint_2 = pixel(T_W_C1, W_f), pixel(T_W_C2, W_f)
    T_C1_C2 = T_W_C1.inv() * T_W_C2
    
    assert epipolar_residual(T_C1_C2, K, keypoint_1, keypoint_2) < 1e-9
    # Moving the keypoint along the epipolar line normal by d pixels gives a Sampson distance of about d/2 (error split over the two images)
    F = np.linalg.inv(K).T @ skew(T_C1_C2.t) @ T_C1_C2.R @ np.linalg.inv(K)
    line = F.T @ np.append(keypoint_1, 1)
    normal = line[:2] / np.linalg.norm(line[:2])
    assert epipolar_residual(T_C1_C2, K, keypoint_1, keypoint_2 + 10.0 * normal) > 2.0


def feature_seen_by(cameras: dict, W_f: np.ndarray):
    """Feature observed without noise by all cameras, anchored to the first one with the exact inverse depth."""
    from src.msckf.FeatureExtractor import Feature
    from src.utils.geometry import InverseDepthPoint
    feature = Feature()
    for index, camera in cameras.items():
        feature.keypoints.append((K @ np.append(project(camera.T_W_Ci, W_f), 1))[:2])
        feature.camera_indices.append(index)
    anchor = cameras[feature.camera_indices[0]].T_W_Ci
    feature.inverse_depth_point = InverseDepthPoint(anchor, W_f - anchor.t)
    feature.inverse_depth_point.update_depth(np.linalg.norm(W_f - anchor.t))
    return feature


def test_feature_residual_and_jacobian():
    """MSCKF.compute_residual_and_jacobians before the null-space projection (observability constraint disabled with g = 0)."""
    T_W_C0 = random_pose()
    poses = [T_W_C0 * Isometry3D(exp_so3(rng.normal(scale=0.05, size=3)), rng.normal(scale=0.3, size=3)) for _ in range(4)]
    cameras = {k: Camera(K=K, width=640, height=480, T_W_Ci=T) for k, T in enumerate(poses)}
    W_f = point_in_front(T_W_C0)
    feature = feature_seen_by(cameras, W_f)
    
    msckf = fake_msckf(random_pose(), Isometry3D(np.eye(3), np.zeros(3)), np.eye(15 + 6 * len(cameras)), cameras=cameras)
    msckf.state.imu.W_gravity = np.zeros(3)
    msckf.project_on_nullspace = lambda H_f, r, H_x: (r, H_x) # keep the raw stacked residual and Jacobian
    r, H_x = MSCKF.compute_residual_and_jacobians(msckf, feature)
    
    np.testing.assert_allclose(r, 0, atol=1e-9, err_msg='residual of a noiseless feature')
    
    # r = z - h(x_est) ≈ H δx with x_true = x_est ⊞ δx, hence H = d h(x_est ⊞ δx) / d δx
    def prediction(d):
        return np.concatenate([project(apply_correction(T, d[6*k:6*k+3], d[6*k+3:6*k+6]), W_f) for k, T in enumerate(poses)])
    
    numerical = numerical_jacobian(prediction, 6 * len(cameras))
    np.testing.assert_allclose(H_x[:, 15:], numerical, atol=1e-6, err_msg='H_x camera blocks')


def test_triangulation_sets_inverse_depth_point():
    """After MSCKF.get_valid_features triangulates a lost feature, its inverse-depth point must be the triangulated point."""
    from src.utils.geometry import Line
    T_W_C0 = random_pose()
    poses = [T_W_C0 * Isometry3D(exp_so3(rng.normal(scale=0.05, size=3)), rng.normal(scale=0.5, size=3)) for _ in range(4)]
    cameras = {k: Camera(K=K, width=640, height=480, T_W_Ci=T) for k, T in enumerate(poses)}
    # Point far from the optical axis, where depth along z and distance along the ray differ the most
    W_f = T_W_C0.transform(np.array([2.5, 1.8, 4.0]))
    feature = feature_seen_by(cameras, W_f)
    feature.inverse_depth_point.update_depth(1.0) # wrong initial guess, to be fixed by the triangulation
    for keypoint, index in zip(feature.keypoints, feature.camera_indices):
        camera = cameras[index]
        feature.lines.append(Line(camera.T_W_Ci.t, camera.Ci2W(camera.inverse_project_point(keypoint), is_versor=True), 1.0))
    feature.tracked_for_n_frames, feature.lost_for_n_frames = len(cameras), 1
    
    msckf = fake_msckf(random_pose(), Isometry3D(np.eye(3), np.zeros(3)), np.eye(15 + 6 * len(cameras)), cameras=cameras)
    msckf.__dict__.update(min_number_of_frames_to_be_lost=1, min_number_of_frames_to_be_tracked=2, use_parallax=False,
                          min_parallax=0, triangulated_points={},
                          max_triangulation_error=1e-3, number_of_features_discarded_for_triangulation=0)
    valid, _ = MSCKF.get_valid_features(msckf, {0: feature})
    
    assert 0 in valid
    idp = feature.inverse_depth_point
    np.testing.assert_allclose(idp.base + idp.m / idp.rho, W_f, atol=1e-6)


def test_projected_jacobian_has_no_yaw_information():
    """
    With the observability constraint, the projected measurement Jacobian must be blind to the unobservable
    rotation about gravity (Hesch et al. 2014): H_o N = 0, where for each camera clone N = [R_C_W g; -[p_C]x g].
    """
    T_W_C0 = random_pose()
    poses = [T_W_C0 * Isometry3D(exp_so3(rng.normal(scale=0.05, size=3)), rng.normal(scale=0.3, size=3)) for _ in range(4)]
    cameras = {k: Camera(K=K, width=640, height=480, T_W_Ci=T) for k, T in enumerate(poses)}
    feature = feature_seen_by(cameras, point_in_front(T_W_C0))
    # Current estimates drifted from the first estimates (null states), as after a few EKF updates
    for camera in cameras.values():
        camera.T_W_Ci = apply_correction(camera.T_W_Ci_null, rng.normal(scale=0.02, size=3), rng.normal(scale=0.05, size=3))
    
    msckf = fake_msckf(random_pose(), Isometry3D(np.eye(3), np.zeros(3)), np.eye(15 + 6 * len(cameras)), cameras=cameras)
    msckf.state.imu.W_gravity = np.array([0, 0, -9.81])
    msckf.project_on_nullspace = lambda H_f, r, H_x: MSCKF.project_on_nullspace(msckf, H_f, r, H_x)
    _, H_o = MSCKF.compute_residual_and_jacobians(msckf, feature)
    
    g = msckf.state.imu.W_gravity
    N = np.zeros(15 + 6 * len(cameras))
    for k, camera in cameras.items(): # unobservable direction evaluated at the first estimates
        N[15 + 6*k: 18 + 6*k] = camera.T_W_Ci_null.R.T @ g
        N[18 + 6*k: 21 + 6*k] = -skew(camera.T_W_Ci_null.t) @ g
    np.testing.assert_allclose(H_o @ N, 0, atol=1e-8)


def test_used_observations_are_discarded():
    """After an update a still-tracked feature keeps only its last observation, so earlier measurements are not reused."""
    T_W_C0 = random_pose()
    poses = [T_W_C0 * Isometry3D(exp_so3(rng.normal(scale=0.05, size=3)), rng.normal(scale=0.3, size=3)) for _ in range(4)]
    cameras = {k: Camera(K=K, width=640, height=480, T_W_Ci=T) for k, T in enumerate(poses)}
    feature = feature_seen_by(cameras, point_in_front(T_W_C0))
    feature.descriptors = [np.full(4, k, dtype=float) for k in range(4)]
    feature.scores = [1.0] * 4
    feature.lines = list(range(4))
    feature.tracked_for_n_frames = 4
    msckf = SimpleNamespace(features={7: feature})
    
    MSCKF.discard_used_observations(msckf, [7])
    
    assert feature.camera_indices == [3] and feature.tracked_for_n_frames == 1
    assert len(feature.keypoints) == len(feature.descriptors) == len(feature.scores) == len(feature.lines) == 1
    np.testing.assert_array_equal(feature.descriptors[0], np.full(4, 3.0))


def test_null_space_projection():
    """The projection removes the feature (A^T H_f = 0) with an orthonormal basis, so the noise stays sigma^2 I."""
    H_f = rng.normal(size=(10, 3))
    H_x = rng.normal(size=(10, 27))
    r = rng.normal(size=(10, 1))
    r_o, H_o = MSCKF.project_on_nullspace(None, H_f, r, H_x)
    assert r_o.shape == (7, 1) and H_o.shape == (7, 27)
    # Recover A from H_o = A^T H_x with H_x of full row rank
    A_T = H_o @ np.linalg.pinv(H_x)
    np.testing.assert_allclose(A_T @ H_f, 0, atol=1e-9)
    np.testing.assert_allclose(A_T @ A_T.T, np.eye(7), atol=1e-9)
    np.testing.assert_allclose(r_o, A_T @ r, atol=1e-9)


@pytest.mark.parametrize('trial', range(5))
def test_triangulation_recovers_point(trial):
    T_W_C0 = random_pose()
    poses = [T_W_C0 * Isometry3D(exp_so3(rng.normal(scale=0.05, size=3)), rng.normal(scale=0.3, size=3)) for _ in range(5)]
    W_f = point_in_front(T_W_C0)
    keypoints = [(K @ np.append(project(T, W_f), 1))[:2] for T in poses]
    initial = T_W_C0.t + 1.4 * (W_f - T_W_C0.t) # 40 % too far
    estimate, rms = triangulate_feature(poses, keypoints, np.linalg.inv(K), initial)
    np.testing.assert_allclose(estimate, W_f, atol=1e-6)
    assert rms < 1e-9


def test_triangulation_rejects_point_behind_cameras():
    T_W_C0 = random_pose()
    poses = [T_W_C0 * Isometry3D(np.eye(3), np.array([0.2 * k, 0, 0])) for k in range(4)]
    W_f = T_W_C0.transform(np.array([0.3, -0.2, -4.0])) # behind
    keypoints = [(K @ np.append(project(T, W_f), 1))[:2] for T in poses]
    estimate, _ = triangulate_feature(poses, keypoints, np.linalg.inv(K), W_f)
    assert estimate is None
