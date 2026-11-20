import os
import cv2
import yaml
import numpy as np
import pandas as pd
from scipy.spatial.transform import Slerp
from src.utils.geometry import *

BASE_DATASET_PATH = './data'

class EurocConverter():
    """
    Converts an EuRoC MAV sequence (ASL format) into the files read by the Parser:
    imu.csv and camera.csv with timestamps in seconds from the first IMU sample, and
    transforms_gt.csv with the ground-truth IMU pose resampled at every IMU timestamp
    (NaN where the ground truth is not available).

    Unlike the photorealistic datasets, IMU data is real: nothing is generated.
    The camera calibration (intrinsics, radial-tangential distortion, camera-IMU extrinsics)
    and the IMU noise parameters are read from the sensor.yaml files.

    Expected layout: data/euroc/<sequence>/mav0/{cam0, imu0, state_groundtruth_estimate0}
    """

    def __init__(self, sequence: str):
        self.root_folder = f'{BASE_DATASET_PATH}/euroc/{sequence}/'
        self.mav_folder = self.root_folder + 'mav0/'

        cam_info = self._read_yaml(self.mav_folder + 'cam0/sensor.yaml')
        imu_info = self._read_yaml(self.mav_folder + 'imu0/sensor.yaml')

        #* Camera
        fu, fv, cu, cv = cam_info['intrinsics']
        self.K_distorted = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]])
        self.distortion = np.array(cam_info['distortion_coefficients'])  # k1, k2, p1, p2 (same order as OpenCV)
        self.width, self.height = cam_info['resolution']

        # Images are undistorted once with these maps, so the filter can keep its pinhole model
        self.K, _ = cv2.getOptimalNewCameraMatrix(self.K_distorted, self.distortion, (self.width, self.height), alpha=0)
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(self.K_distorted, self.distortion, None, self.K,
                                                             (self.width, self.height), cv2.CV_32FC1)

        #* Camera-IMU extrinsics: T_BS of each sensor is its pose in the body frame
        T_B_C = np.array(cam_info['T_BS']['data']).reshape(4, 4)
        T_B_I = np.array(imu_info['T_BS']['data']).reshape(4, 4)
        T_I_C = np.linalg.inv(T_B_I) @ T_B_C
        self.T_I_C = Isometry3D(T_I_C[:3, :3], T_I_C[:3, 3])

        #* IMU noise parameters (continuous time)
        self.accelerometer_noise_density = imu_info['accelerometer_noise_density']
        self.gyroscope_noise_density = imu_info['gyroscope_noise_density']
        self.accelerometer_random_walk = imu_info['accelerometer_random_walk']
        self.gyroscope_random_walk = imu_info['gyroscope_random_walk']

    @staticmethod
    def _read_yaml(path: str) -> dict:
        with open(path) as f: return yaml.safe_load(f)

    def undistort(self, image: np.ndarray) -> np.ndarray:
        return cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR)

    def process_data(self):
        #* IMU
        imu = pd.read_csv(self.mav_folder + 'imu0/data.csv')
        imu.columns = ['timestamp_ns', 'wx', 'wy', 'wz', 'ax', 'ay', 'az']
        t0_ns = imu['timestamp_ns'].iloc[0]
        imu_timestamps = (imu['timestamp_ns'] - t0_ns).values * 1e-9

        imu_out = pd.DataFrame({'timestamp': imu_timestamps,
                                'wx': imu['wx'], 'wy': imu['wy'], 'wz': imu['wz'],
                                'ax': imu['ax'], 'ay': imu['ay'], 'az': imu['az']})
        imu_out.to_csv(self.root_folder + 'imu.csv', index=False)

        #* Camera
        cam = pd.read_csv(self.mav_folder + 'cam0/data.csv')
        cam.columns = ['timestamp_ns', 'filename']
        cam = cam[cam['timestamp_ns'] >= t0_ns]
        cam_out = pd.DataFrame({'timestamp': (cam['timestamp_ns'] - t0_ns).values * 1e-9,
                                'image_path': [self.mav_folder + 'cam0/data/' + f for f in cam['filename']]})
        cam_out.to_csv(self.root_folder + 'camera.csv', index=False)

        #* Ground truth: IMU pose in the Vicon world frame, resampled at the IMU timestamps
        gt = pd.read_csv(self.mav_folder + 'state_groundtruth_estimate0/data.csv')
        gt_timestamps = (gt.iloc[:, 0] - t0_ns).values * 1e-9
        gt_positions = gt.iloc[:, 1:4].values
        gt_quaternions = gt.iloc[:, [5, 6, 7, 4]].values  # w, x, y, z -> x, y, z, w

        valid = (imu_timestamps >= gt_timestamps[0]) & (imu_timestamps <= gt_timestamps[-1])
        positions = np.full((len(imu_timestamps), 3), np.nan)
        rotations = np.full((len(imu_timestamps), 3, 3), np.nan)
        for k in range(3):
            positions[valid, k] = np.interp(imu_timestamps[valid], gt_timestamps, gt_positions[:, k])
        rotations[valid] = Slerp(gt_timestamps, scipyR.from_quat(gt_quaternions))(imu_timestamps[valid]).as_matrix()

        gt_out = pd.DataFrame({'timestamp': imu_timestamps,
                               'T00': rotations[:, 0, 0], 'T01': rotations[:, 0, 1], 'T02': rotations[:, 0, 2], 'T03': positions[:, 0],
                               'T10': rotations[:, 1, 0], 'T11': rotations[:, 1, 1], 'T12': rotations[:, 1, 2], 'T13': positions[:, 1],
                               'T20': rotations[:, 2, 0], 'T21': rotations[:, 2, 1], 'T22': rotations[:, 2, 2], 'T23': positions[:, 2]})
        gt_out.to_csv(self.root_folder + 'transforms_gt.csv', index=False)
