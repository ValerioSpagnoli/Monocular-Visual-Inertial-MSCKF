"""IMU model checks with the physical accelerometer convention f = R_W_I^T (a - g), g = [0, 0, -9.81]."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.utils.geometry import Isometry3D
from src.msckf.IMU import IMU, IMUMeasurement

G = np.array([0, 0, -9.81])


def specific_force(R_W_I: np.ndarray, a_W: np.ndarray = np.zeros(3)) -> np.ndarray:
    return R_W_I.T @ (a_W - G)


def test_static_imu_does_not_move():
    R_W_I = Rotation.random(random_state=1).as_matrix()
    imu = IMU(T_W_Ii=Isometry3D(R_W_I, np.zeros(3)), v_W_Ii=np.zeros(3), W_gravity=G)
    for _ in range(200):
        imu.integrate(linear_acceleration=specific_force(R_W_I), angular_velocity=np.zeros(3), dt=0.005)
    np.testing.assert_allclose(imu.v_W_Ii, 0, atol=1e-9)
    np.testing.assert_allclose(imu.T_W_Ii.t, 0, atol=1e-9)


def test_constant_acceleration():
    R_W_I = Rotation.random(random_state=2).as_matrix()
    a_W = np.array([0.5, -0.2, 0.1])
    imu = IMU(T_W_Ii=Isometry3D(R_W_I, np.zeros(3)), v_W_Ii=np.zeros(3), W_gravity=G)
    dt, steps = 0.005, 200
    for _ in range(steps):
        imu.integrate(linear_acceleration=specific_force(R_W_I, a_W), angular_velocity=np.zeros(3), dt=dt)
    T = dt * steps
    np.testing.assert_allclose(imu.v_W_Ii, a_W * T, atol=1e-9)
    np.testing.assert_allclose(imu.T_W_Ii.t, 0.5 * a_W * T**2, atol=1e-9)


@pytest.mark.parametrize('R_W_I', [np.eye(3),                                          # already level (theta = 0)
                                   np.diag([1.0, -1.0, -1.0]),                         # upside down (theta = pi)
                                   Rotation.random(random_state=3).as_matrix(),
                                   Rotation.from_euler('y', 90, degrees=True).as_matrix()]) # x axis up, as on EuRoC
def test_initialization_recovers_tilt(R_W_I):
    measurements = [IMUMeasurement(timestamp=0.005 * k, angular_velocity=np.zeros(3), linear_acceleration=specific_force(R_W_I))
                    for k in range(10)]
    imu = IMU(W_gravity=G)
    imu.initialize(measurements)
    R = imu.T_W_Ii.R
    assert np.all(np.isfinite(R))
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-9)
    # Gravity seen from the body must match: only the yaw about the vertical is arbitrary
    np.testing.assert_allclose(R.T @ G, R_W_I.T @ G, atol=1e-9)
