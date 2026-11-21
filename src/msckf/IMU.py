from dataclasses import dataclass, field
from typing import List
import numpy as np
from scipy.spatial.transform import Rotation
from src.utils.geometry import *

@dataclass
class IMUMeasurement:
    timestamp: float = 0.0
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    linear_acceleration: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    gravity: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

class IMU:
    def __init__(self,
                T_W_I: Isometry3D = Isometry3D(np.eye(3), np.zeros(3)),
                T_W_Ii: Isometry3D = Isometry3D(np.eye(3), np.zeros(3)),   
                v_W_Ii: np.ndarray = np.zeros(3),
                W_gravity: np.ndarray = np.array([0, 0, -9.81]),
                id: int = 0,
                timestamp: float = 0.0
        ) -> None:

        self.id = id
        self.timestamp = timestamp
        
        self.T_W_I = T_W_I    # static transform of IMU Frame in World Frame
        self.T_W_Ii = T_W_Ii  # transform of IMU Frame i-th in World Frame
        self.v_W_Ii = v_W_Ii  # velocity of the IMU frame in world frame at time i
        self.w_W_Ii = np.zeros(3) # angular velocity of the IMU frame in world frame at time i
        self.a_W_Ii = np.zeros(3) # linear acceleration of the IMU frame in world frame at time i
        self.dt = 0.0
        
        self.gyroscope_bias = np.zeros(3)
        self.accelerometer_bias = np.zeros(3) 
        self.W_gravity = W_gravity
        self.planet_angular_velocity = np.array([0, 0, 0])
        
        self.T_W_Ii_null = Isometry3D(np.eye(3), np.zeros(3))
        self.v_W_Ii_null = np.zeros(3)
        
        self.is_initialized = False
        
    def initialize(self, imu_measurements: List[IMUMeasurement], estimate_gyroscope_bias: bool = False):
        self.is_initialized = True
        
        mean_angular_velocity = np.mean([imu_measurement.angular_velocity for imu_measurement in imu_measurements], axis=0)
        mean_linear_acceleration = np.mean([imu_measurement.linear_acceleration for imu_measurement in imu_measurements], axis=0)
        
        # At rest the accelerometer measures f = R_W_I^T (0 - g): the specific force points up, opposite to gravity.
        # R_W_I is the smallest rotation taking the measured up direction onto the world up direction (yaw is arbitrary).
        W_up = -self.W_gravity / np.linalg.norm(self.W_gravity)
        I_up = mean_linear_acceleration / np.linalg.norm(mean_linear_acceleration)
        R_W_I = Rotation.align_vectors(W_up[None], I_up[None])[0].as_matrix()
        
        self.T_W_Ii = Isometry3D(R_W_I, np.zeros(3))
        
        # Valid only if the platform is static during the initialization window
        if estimate_gyroscope_bias: self.gyroscope_bias = mean_angular_velocity
        # I_gravity = R_W_I.T @ self.W_gravity
        # self.accelerometer_bias = mean_linear_acceleration - I_gravity

            
        # print(f'IMU initialized')
        # print(f'mean_accelerometer: {mean_linear_acceleration}')
        # print(f'mean_gyroscope:     {mean_angular_velocity}')
        # print(f'I_Gravity dir:      {I_gravity / np.linalg.norm(I_gravity)}')
        # print(f'I_Gravity:          {I_gravity}')
        # print(f'W_Gravity:          {self.W_gravity}')
        # print(f'Accelerometer bias: {self.accelerometer_bias}')
        # print(f'Gyroscope bias:     {self.gyroscope_bias}')
        # print(f'R_W_I:\n{self.T_W_Ii.R}')
        # print(f'-----------------------------------------------\n')
        
    def integrate(self, linear_acceleration: np.ndarray, angular_velocity: np.ndarray, dt: float):
        
        p_W_Ii = self.T_W_Ii.t
        R_W_Ii = self.T_W_Ii.R
        
        angular_velocity = angular_velocity - self.T_W_Ii.R.T @ self.planet_angular_velocity
        theta = np.linalg.norm(angular_velocity) * dt   
        if theta > 0:   
            axis = angular_velocity / np.linalg.norm(angular_velocity)
            skew_axis = skew(axis)
            R_W_Ii_dot = np.eye(3) + np.sin(theta) * skew_axis + (1 - np.cos(theta)) * skew_axis @ skew_axis
        else:
            R_W_Ii_dot = np.eye(3)
            
        R_W_Ii_new = R_W_Ii @ R_W_Ii_dot
        
        # The accelerometer measures the specific force f = R_W_I^T (a - g)
        linear_acceleration = R_W_Ii @ linear_acceleration + self.W_gravity
                    
        p_W_Ii_new = p_W_Ii + self.v_W_Ii * dt + 0.5 * linear_acceleration * dt**2
        v_W_Ii_new = self.v_W_Ii + linear_acceleration * dt
        
        self.T_W_Ii = Isometry3D(R_W_Ii_new, p_W_Ii_new)
        self.v_W_Ii = v_W_Ii_new