from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as scipyR

@dataclass
class Line:
    """
    Line in 3D space.
    
    Attributes:
        base (numpy.ndarray): Base point of the line.
        direction (numpy.ndarray): Direction of the line (versor).
        confidence (float): Confidence of the line.
    """
    
    base: np.ndarray = np.zeros(3)
    direction: np.ndarray = np.zeros(3)
    confidence: float = 1.0
        
    def __str__(self) -> str:    
        return f'Base: {self.base} - Direction: {self.direction} - Confidence: {self.confidence}'

class Isometry3D():  
    def __init__(self, R: np.ndarray = np.eye(3), t: np.ndarray = np.zeros(3)):    
        self.R = R
        self.t = t.reshape((1,3))[0]
                        
    def __str__(self) -> str:
        return f'R:\n{self.R}\nt: {self.t}'
    
    def __mul__(self, other: 'Isometry3D') -> 'Isometry3D':
        I = self.matrix() @ other.matrix()
        return Isometry3D(I[:3,:3], I[:3,3])
    
    def inv(self) -> 'Isometry3D':
        I_inv = np.linalg.inv(self.matrix())
        return Isometry3D(I_inv[:3,:3], I_inv[:3,3])
    
    def T(self) -> 'Isometry3D':
        I_transpose = self.matrix().T
        return Isometry3D(I_transpose[:3,:3], I_transpose[:3,3])
        
    def transform(self, p: np.ndarray, rotation_only=False) -> np.ndarray:
        if rotation_only: return self.R @ p
        else: return self.R @ p + self.t
        
    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3,:3] = self.R
        T[:3,3] = self.t
        return T

class InverseDepthPoint():
    def __init__(self, camera_pose: Isometry3D = Isometry3D(np.eye(3), np.zeros(3)), direction: np.ndarray = np.zeros(3)) -> None:
        self.base = camera_pose.t
        self.theta = np.arctan2(direction[0], direction[2])
        self.phi = np.arctan2(-direction[1], np.sqrt(direction[0]**2 + direction[2]**2))
        self.m = np.array([np.cos(self.phi)*np.sin(self.theta), -np.sin(self.phi), np.cos(self.phi)*np.cos(self.theta)]).T
        self.rho = 0.1
                            
    def update_depth(self, depth: float) -> None:
        self.rho = 1/depth

    def update_m(self, direction: np.ndarray) -> None:
        self.theta = np.arctan2(direction[0], direction[2])
        self.phi = np.arctan2(-direction[1], np.sqrt(direction[0]**2 + direction[2]**2))
        self.m = np.array([np.cos(self.phi)*np.sin(self.theta), -np.sin(self.phi), np.cos(self.phi)*np.cos(self.theta)]).T
        
    def update(self, depth: float, direction: np.ndarray) -> None:
        self.update_depth(depth)
        self.update_m(direction)
    

def Rx(theta: float) -> np.ndarray:
    """
    Rotation matrix around x-axis.
    
    Args:
        theta (float): Angle.
        
    Returns:
        numpy.ndarray: Rotation matrix around x-axis.
    """
    
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])    
    
def Ry(theta: float) -> np.ndarray:
    """
    Rotation matrix around y-axis.
    
    Args:
        theta (float): Angle.
        
    Returns:
        numpy.ndarray: Rotation matrix around y-axis.
    """
    
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def Rz(theta: float) -> np.ndarray:
    """
    Rotation matrix around z-axis.
    
    Args:
        theta (float): Angle.
        
    Returns:
        numpy.ndarray: Rotation matrix around z-axis.
    """
    
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])
    
def euler2R(euler: np.ndarray, intrinsic: bool=True) -> np.ndarray:
    """
    Euler angles to Rotation matrix.
    
    Args:
        euler (numpy.ndarray): Euler angles.
        
    Returns:
        numpy.ndarray: Rotation matrix
    """
    
    return scipyR.from_euler('XYZ' if intrinsic else 'xyz', euler).as_matrix()

def R2euler(R: np.ndarray, intrinsic: bool=True) -> np.ndarray:
    """
    Rotation matrix to Euler angles.
    
    Args:
        R (numpy.ndarray): Rotation matrix.
        
    Returns:
        numpy.ndarray: Euler angles.
    """
    
    return scipyR.from_matrix(R).as_euler('XYZ' if intrinsic else 'xyz')

def euler2quaternion(euler: np.ndarray, intrinsic: bool=True) -> np.ndarray:
    """
    Quaternion from Euler angles.
    
    Args:
        euler (numpy.ndarray): Euler angles.
        
    Returns:
        numpy.ndarray: Quaternion x,y,z,w.
    """
    return scipyR.from_euler('XYZ' if intrinsic else 'xyz', euler).as_quat()

def quaternion2euler(q: np.ndarray, intrinsic: bool=True) -> np.ndarray:
    """
    Euler angles from quaternion.
    
    Args:
        q (numpy.ndarray): Quaternion.
        
    Returns:
        numpy.ndarray: Euler angles.
    """
    
    return scipyR.from_quat(q).as_euler('XYZ' if intrinsic else 'xyz')

def quaternion2R(q: np.ndarray) -> np.ndarray:
    """
    Rotation matrix from quaternion.
    
    Args:
        q (numpy.ndarray): Quaternion.
        
    Returns:
        numpy.ndarray: Rotation matrix.
    """
    
    return scipyR.from_quat(q).as_matrix()

def R2quaternion(R: np.ndarray) -> np.ndarray:
    """
    Quaternion from Rotation matrix.
    
    Args:
        R (numpy.ndarray): Rotation matrix.
        
    Returns:
        numpy.ndarray: Quaternion.
    """
    
    return scipyR.from_matrix(R).as_quat()

def R2axisAngle(R: np.ndarray) -> np.ndarray:
    """
    Axis-angle from Rotation matrix.
    
    Args:
        R (numpy.ndarray): Rotation matrix.
        
    Returns:
        numpy.ndarray: Axis-angle.
    """
    
    return scipyR.from_matrix(R).as_rotvec()

def axisAngle2R(axis_angle: np.ndarray) -> np.ndarray:
    """
    Rotation matrix from axis-angle.
    
    Args:
        axis_angle (numpy.ndarray): Axis-angle.
        
    Returns:
        numpy.ndarray: Rotation matrix.
    """
    
    return scipyR.from_rotvec(axis_angle).as_matrix()

def skew(w : np.ndarray) -> np.ndarray:
    """
    Skew-symmetric matrix.
    
    Args:
        w (numpy.ndarray): Vector 3D.
        
    Returns:
        numpy.ndarray: Skew-symmetric matrix.
    """
    
    return np.array([[    0, -w[2],  w[1]], 
                     [ w[2],     0, -w[0]], 
                     [-w[1],  w[0],    0]])

def angle_between_directions(d1: np.ndarray, d2: np.ndarray) -> float:
    """
    Angle between two directions.
    
    Args:
        direction_1 (numpy.ndarray): Direction 1.
        direction_2 (numpy.ndarray): Direction 2.
        
    Returns:
        float: Angle between the two directions.
    """
    
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    
    dot_product = np.dot(d1, d2)        
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    alpha = np.arccos(dot_product)       
    return alpha 

def white_gaussian_noise(sigma: list[float]) -> np.ndarray:
    """
    Generate white Gaussian noise with given standard deviations.
    Parameters:
        sigma (list[float]): A list of standard deviations for each dimension.
    Returns:
        np.ndarray: A numpy array containing the generated noise.
        np.ndarray: The covariance matrix used to generate the noise.
    """
    
    mean = np.zeros(len(sigma))
    covariance = np.diag(sigma)**2
    noise = np.random.multivariate_normal(mean, covariance)

    return noise, covariance
    
def intersection_of_lines(lines: list[Line]) -> np.ndarray:
    """
    Intersection of lines.
    
    Args:
        lines (list[Line]): List of lines.  
        
    Returns:
        numpy.ndarray: Intersection point.
    """
    
    X = np.zeros((3, 3))
    y = np.zeros((3,1))
    
    for line in lines:
        base = np.reshape(line.base, (3, 1))
        direction = np.reshape(line.direction, (3, 1))
        direction = direction / np.linalg.norm(direction)
        confidence = line.confidence
        
        P = np.eye(3) - (direction @ direction.T)
        
        X += confidence * P
        y += confidence * P @ base
    
    b = np.linalg.pinv(X) @ y
    # C = np.linalg.inv(X.T @ X)
    C = np.eye(3)
    
    return np.reshape(b, (3,)), C