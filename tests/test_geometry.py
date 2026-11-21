"""Basic checks of the geometry helpers."""
import numpy as np
from scipy.spatial.transform import Rotation

from src.utils.geometry import Isometry3D, InverseDepthPoint, Line, intersection_of_lines, skew
from src.msckf.IMU import IMU


def test_isometry_instances_do_not_share_state():
    a, b = Isometry3D(), Isometry3D()
    a.t += 1.0
    a.R[0, 0] = 5.0
    np.testing.assert_array_equal(b.t, np.zeros(3))
    np.testing.assert_array_equal(b.R, np.eye(3))
    
    t = np.ones(3)
    c = Isometry3D(np.eye(3), t)
    t[0] = 7.0
    assert c.t[0] == 1.0


def test_imu_instances_do_not_share_state():
    a, b = IMU(), IMU()
    a.T_W_Ii.t[0] = 3.0
    a.v_W_Ii[0] = 3.0
    assert b.T_W_Ii.t[0] == 0.0 and b.v_W_Ii[0] == 0.0


def test_isometry_composition_and_inverse():
    T = Isometry3D(Rotation.random(random_state=0).as_matrix(), np.array([1.0, -2.0, 0.5]))
    I = T * T.inv()
    np.testing.assert_allclose(I.R, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(I.t, 0, atol=1e-12)
    p = np.array([0.3, 0.2, -1.0])
    np.testing.assert_allclose(T.inv().transform(T.transform(p)), p, atol=1e-12)


def test_skew_is_cross_product():
    a, b = np.array([1.0, 2.0, 3.0]), np.array([-0.5, 0.4, 2.0])
    np.testing.assert_allclose(skew(a) @ b, np.cross(a, b))


def test_intersection_of_lines():
    point = np.array([1.0, 2.0, 5.0])
    bases = [np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0.5])]
    lines = [Line(base, point - base, 1.0) for base in bases]
    estimate, _ = intersection_of_lines(lines)
    np.testing.assert_allclose(estimate, point, atol=1e-9)


def test_inverse_depth_point():
    camera = Isometry3D(np.eye(3), np.array([1.0, 0, 0]))
    point = np.array([3.0, -1.0, 4.0])
    idp = InverseDepthPoint(camera, point - camera.t)
    idp.update_depth(np.linalg.norm(point - camera.t))
    np.testing.assert_allclose(idp.base + idp.m / idp.rho, point, atol=1e-12)
