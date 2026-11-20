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
from src.msckf.MSCKF import MSCKF, MSCKFState

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
