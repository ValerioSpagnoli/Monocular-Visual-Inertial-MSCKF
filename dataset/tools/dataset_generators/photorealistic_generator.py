import pandas as pd
import numpy as np
import tqdm as tqdm
import os
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Slerp
from src.utils.geometry import *

np.random.seed(42)

BASE_DATASET_PATH = './data'

class PhotorealisticGenerator():
    def __init__(self, source: str, sequence: str,
                 accelerometer_noise_density: float = 0.01, 
                 gyroscope_noise_density: float = 0.001, 
                 accelerometer_random_walk: float = 0.001, 
                 gyroscope_random_walk: float = 0.0001):
        
        self.root_folder = f'{BASE_DATASET_PATH}/{source}/{sequence}/'
        
        self.gt_trajectory_file = self.root_folder + 'trajectory.csv'
        if source == 'peringlab': 
            self.gt_trajectory_columns = ['timestamp', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']
            self.gt_trajectory_separator = ','
        elif source == 'tartanair': 
            self.gt_trajectory_columns = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']   
            self.gt_trajectory_separator = ' '
            
        self.gt_imu_file = self.root_folder + 'imu_gt.csv'
        self.cam_files = self.root_folder + 'cam'
        self.noisy_imu_file = self.root_folder + 'imu.csv'

        camera_info = pd.read_csv(f'{BASE_DATASET_PATH}/{source}/camera_info.csv')
        # Pose of the frame used by the trajectory file in the IMU frame.
        # TartanAir poses are given in the NED camera frame (x forward, y right, z down), which is taken as IMU frame;
        # for the other sources the trajectory is the camera (optical) pose and camera_info.csv gives its rotation in the IMU frame.
        self.trajectory_frame_is_imu = source == 'tartanair'
        self.T_W_C = np.eye(4)
        self.T_W_C[0:3, 0:3] = np.array([[camera_info.iloc[0]['R00'], camera_info.iloc[0]['R01'], camera_info.iloc[0]['R02']],
                                         [camera_info.iloc[0]['R10'], camera_info.iloc[0]['R11'], camera_info.iloc[0]['R12']],
                                         [camera_info.iloc[0]['R20'], camera_info.iloc[0]['R21'], camera_info.iloc[0]['R22']]])

        self.accelerometer_noise_density = accelerometer_noise_density
        self.gyroscope_noise_density = gyroscope_noise_density
        self.accelerometer_random_walk = accelerometer_random_walk
        self.gyroscope_random_walk = gyroscope_random_walk
        self.W_gravity = np.array([0, 0, -9.81])
        self.imu_samples_per_frame = 9
            
    def process_data(self):
        #* Process Camera data
        gt_cam_paths = sorted(os.listdir(self.cam_files))
        
        cam_rows_list = []
        for i in range(3):
            timestamp = i*0.05
            image_path = f'{self.cam_files}/' + gt_cam_paths[i] 
            cam_rows_list.append({'timestamp': timestamp, 'image_path': image_path})
        
        timestamp_offset = timestamp
        for i in range(1, len(gt_cam_paths)):
            timestamp = i*0.05 + timestamp_offset
            image_path = f'{self.cam_files}/' + gt_cam_paths[i]
            cam_rows_list.append({'timestamp': timestamp, 'image_path': image_path})
        
        gt_cam = pd.DataFrame(cam_rows_list)
        gt_cam.to_csv(f'{self.root_folder}/camera.csv', index=False)
        
        
        #* Process IMU data
        gt_trajectory = pd.read_csv(self.gt_trajectory_file, sep=self.gt_trajectory_separator)
        gt_trajectory.columns = self.gt_trajectory_columns
        
        T_W_C0 = np.eye(4)
        T_W_C0[0:3, 3] = np.array([gt_trajectory.iloc[0]['x'], gt_trajectory.iloc[0]['y'], gt_trajectory.iloc[0]['z']])
        T_W_C0[0:3, 0:3] = quaternion2R([gt_trajectory.iloc[0]['qx'], gt_trajectory.iloc[0]['qy'], gt_trajectory.iloc[0]['qz'], gt_trajectory.iloc[0]['qw']])
        
        T_W_W0 = np.eye(4)
        T_W_W0 = T_W_C0 if self.trajectory_frame_is_imu else T_W_C0 @ np.linalg.inv(self.T_W_C)

        rows_list = []
        for i in range(3):
            timestamp = i*0.05
            rows_list.append({'timestamp': timestamp,
                              'T00': 1.0, 'T01': 0.0, 'T02': 0.0, 'T03': 0.0,
                              'T10': 0.0, 'T11': 1.0, 'T12': 0.0, 'T13': 0.0,
                              'T20': 0.0, 'T21': 0.0, 'T22': 1.0, 'T23': 0.0})
        
        timestamp_offset = timestamp 
        for i in range(1, len(gt_trajectory)):
            row = gt_trajectory.iloc[i]
            
            timestamp = i*0.05 + timestamp_offset
            x = row['x']
            y = row['y']
            z = row['z']
            qw = row['qw']
            qx = row['qx']
            qy = row['qy']
            qz = row['qz']
            
            T_W_Ci = np.eye(4)
            T_W_Ci[0:3, 0:3] = quaternion2R([qx, qy, qz, qw])
            T_W_Ci[0:3, 3] = np.array([x, y, z])
            
            T_W_Wi = T_W_Ci if self.trajectory_frame_is_imu else T_W_Ci @ np.linalg.inv(self.T_W_C)
            T_W0_Wi = np.linalg.inv(T_W_W0) @ T_W_Wi

            new_row = {'timestamp': timestamp,
                       'T00': T_W0_Wi[0, 0], 'T01': T_W0_Wi[0, 1], 'T02': T_W0_Wi[0, 2], 'T03': T_W0_Wi[0, 3],
                       'T10': T_W0_Wi[1, 0], 'T11': T_W0_Wi[1, 1], 'T12': T_W0_Wi[1, 2], 'T13': T_W0_Wi[1, 3],
                       'T20': T_W0_Wi[2, 0], 'T21': T_W0_Wi[2, 1], 'T22': T_W0_Wi[2, 2], 'T23': T_W0_Wi[2, 3]}
            rows_list.append(new_row)
        
        gt_trajectory_transforms = pd.DataFrame(rows_list)
        

        #* Smooth trajectory through the keyframes, sampled at the IMU rate (9 samples per camera interval)
        key_timestamps = gt_trajectory_transforms['timestamp'].values
        key_positions = gt_trajectory_transforms[['T03', 'T13', 'T23']].values
        key_rotations = gt_trajectory_transforms[['T00', 'T01', 'T02', 'T10', 'T11', 'T12', 'T20', 'T21', 'T22']].values.reshape(-1, 3, 3)
        
        timestamps = np.concatenate([np.linspace(t0, t1, self.imu_samples_per_frame + 1)[:-1] for t0, t1 in zip(key_timestamps[:-1], key_timestamps[1:])]
                                    + [key_timestamps[-1:]])
        # PCHIP (C1, no overshoot) keeps the static initial segment exactly static, which the gravity initialization needs:
        # a cubic spline rings there because TartanAir sequences start already moving. Rotations use slerp (constant
        # angular velocity between keyframes).
        position_spline = PchipInterpolator(key_timestamps, key_positions)
        rotation_spline = Slerp(key_timestamps, scipyR.from_matrix(key_rotations))
        velocities = position_spline.derivative()(timestamps)
        rotations = rotation_spline(timestamps).as_matrix()
        
        #* Noise-free IMU measurements that the filter integration reproduces exactly (IMU.integrate):
        #* R_{k+1} = R_k exp(w_k dt), p_{k+1} = p_k + v_k dt + a_k dt^2 / 2, v_{k+1} = v_k + a_k dt, a_k = R_k f_k + g.
        #* Velocities are those of the spline and a_k = (v_k+1 - v_k) / dt; the ground-truth positions are the ones this
        #* scheme reproduces (trapezoidal rule), which stay within a few millimeters of the spline.
        #* The measurement driving the interval [t_k, t_k+1] is stored at t_k+1, as consumed by MSCKF.process_imu.
        angular_velocities = np.zeros((len(timestamps), 3))
        specific_forces = np.zeros((len(timestamps), 3))
        positions = np.zeros((len(timestamps), 3))
        positions[0] = position_spline(timestamps[0])
        for k in range(len(timestamps) - 1):
            dt = timestamps[k + 1] - timestamps[k]
            angular_velocities[k + 1] = scipyR.from_matrix(rotations[k].T @ rotations[k + 1]).as_rotvec() / dt
            acceleration = (velocities[k + 1] - velocities[k]) / dt
            positions[k + 1] = positions[k] + 0.5 * (velocities[k] + velocities[k + 1]) * dt
            specific_forces[k + 1] = rotations[k].T @ (acceleration - self.W_gravity)
        
        #* Noisy measurements: white noise and accumulated bias random walk.
        #* Continuous-time densities -> per-sample values, matching the filter model (Q = Phi G Qc G^T Phi^T dt):
        #* white noise has std sigma / sqrt(dt), the bias random walk grows by sigma * sqrt(dt) per sample
        noisy_angular_velocities = angular_velocities.copy()
        noisy_specific_forces = specific_forces.copy()
        accelerometer_bias = np.zeros(3)
        gyroscope_bias = np.zeros(3)
        for k in range(1, len(timestamps)):
            dt = timestamps[k] - timestamps[k - 1]
            accelerometer_bias += self.accelerometer_random_walk * np.sqrt(dt) * np.random.normal(0, 1, 3)
            gyroscope_bias += self.gyroscope_random_walk * np.sqrt(dt) * np.random.normal(0, 1, 3)
            noisy_specific_forces[k] += accelerometer_bias + self.accelerometer_noise_density / np.sqrt(dt) * np.random.normal(0, 1, 3)
            noisy_angular_velocities[k] += gyroscope_bias + self.gyroscope_noise_density / np.sqrt(dt) * np.random.normal(0, 1, 3)
        
        def imu_table(w, f):
            return pd.DataFrame({'timestamp': timestamps, 'wx': w[:, 0], 'wy': w[:, 1], 'wz': w[:, 2], 'ax': f[:, 0], 'ay': f[:, 1], 'az': f[:, 2]})
        
        imu_table(angular_velocities, specific_forces).to_csv(f'{self.gt_imu_file}', index=False)
        imu_table(noisy_angular_velocities, noisy_specific_forces).to_csv(f'{self.noisy_imu_file}', index=False)
        
        trajectory = {'timestamp': timestamps}
        for r in range(3):
            for c in range(3): trajectory[f'T{r}{c}'] = rotations[:, r, c]
            trajectory[f'T{r}3'] = positions[:, r]
        pd.DataFrame(trajectory)[['timestamp', 'T00', 'T01', 'T02', 'T03', 'T10', 'T11', 'T12', 'T13', 'T20', 'T21', 'T22', 'T23']] \
            .to_csv(f'{self.root_folder}/transforms_gt.csv', index=False)
