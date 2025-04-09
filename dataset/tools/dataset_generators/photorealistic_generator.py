import pandas as pd
import numpy as np
import tqdm as tqdm
import os
from scipy.spatial.transform import Slerp
from src.utils.geometry import *

np.random.seed(42)

BASE_DATASET_PATH = '/home/valeriospagnoli/Thesis/vio/dataset'

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
        self.T_W_C = np.eye(4)
        self.T_W_C[0:3, 0:3] = np.array([[camera_info.iloc[0]['R00'], camera_info.iloc[0]['R01'], camera_info.iloc[0]['R02']],
                                         [camera_info.iloc[0]['R10'], camera_info.iloc[0]['R11'], camera_info.iloc[0]['R12']],
                                         [camera_info.iloc[0]['R20'], camera_info.iloc[0]['R21'], camera_info.iloc[0]['R22']]])

        self.accelerometer_noise_density = accelerometer_noise_density
        self.gyroscope_noise_density = gyroscope_noise_density
        self.accelerometer_random_walk = accelerometer_random_walk
        self.gyroscope_random_walk = gyroscope_random_walk
        self.W_gravity = np.array([0, 0, -9.81])
            
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
        T_W_W0 = T_W_C0 @ np.linalg.inv(self.T_W_C)

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
            
            T_W_Wi = T_W_Ci @ np.linalg.inv(self.T_W_C)
            T_W0_Wi = np.linalg.inv(T_W_W0) @ T_W_Wi

            new_row = {'timestamp': timestamp,
                       'T00': T_W0_Wi[0, 0], 'T01': T_W0_Wi[0, 1], 'T02': T_W0_Wi[0, 2], 'T03': T_W0_Wi[0, 3],
                       'T10': T_W0_Wi[1, 0], 'T11': T_W0_Wi[1, 1], 'T12': T_W0_Wi[1, 2], 'T13': T_W0_Wi[1, 3],
                       'T20': T_W0_Wi[2, 0], 'T21': T_W0_Wi[2, 1], 'T22': T_W0_Wi[2, 2], 'T23': T_W0_Wi[2, 3]}
            rows_list.append(new_row)
        
        gt_trajectory_transforms = pd.DataFrame(rows_list)
        

        trajectory_rows_list = []
        imu_rows_list = []
        imu_noisy_rows_list = []
                    
        first_row = gt_trajectory_transforms.iloc[0]
        first_timestamp = first_row['timestamp']
        first_position = np.array([first_row['T03'], first_row['T13'], first_row['T23']])
        first_rotation = np.array([[first_row['T00'], first_row['T01'], first_row['T02']],
                                   [first_row['T10'], first_row['T11'], first_row['T12']],
                                   [first_row['T20'], first_row['T21'], first_row['T22']]])
        
        trajectory_rows_list.append({'timestamp': first_timestamp,
                                    'T00': first_rotation[0, 0], 'T01': first_rotation[0, 1], 'T02': first_rotation[0, 2], 'T03': first_position[0],
                                    'T10': first_rotation[1, 0], 'T11': first_rotation[1, 1], 'T12': first_rotation[1, 2], 'T13': first_position[1],
                                    'T20': first_rotation[2, 0], 'T21': first_rotation[2, 1], 'T22': first_rotation[2, 2], 'T23': first_position[2]})
        
        imu_rows_list.append({'timestamp': first_timestamp, 
                                'wx': 0, 'wy': 0, 'wz': 0,
                                'ax': 0, 'ay': 0, 'az': 0})
        
        imu_noisy_rows_list.append({'timestamp': first_timestamp,
                                    'wx': 0, 'wy': 0, 'wz': 0,
                                    'ax': 0, 'ay': 0, 'az': 0})
        
        prev_timestamp = first_timestamp
        prev_position = first_position
        prev_rotation = first_rotation
        prev_velocity = np.zeros(3)
        
        accelerometer_bias = np.zeros(3)
        gyroscope_bias = np.zeros(3)
        
        for i in range(1, len(gt_trajectory_transforms)):
                        
            current_row = gt_trajectory_transforms.iloc[i]
            current_timestamp = current_row['timestamp']
            current_position = np.array([current_row['T03'], current_row['T13'], current_row['T23']])
            current_rotation = np.array([[current_row['T00'], current_row['T01'], current_row['T02']],
                                         [current_row['T10'], current_row['T11'], current_row['T12']],
                                         [current_row['T20'], current_row['T21'], current_row['T22']]])
            
            t = np.linspace(0, 1, 10)
            interpolated_timestamps = (np.outer(1 - t, prev_timestamp) + np.outer(t, current_timestamp)).flatten()
            
            interpolated_positions = np.outer(1 - t, prev_position) + np.outer(t, current_position)
                           
            slerp = Slerp([0, 1], scipyR.from_matrix([prev_rotation, current_rotation]))
            interpolated_rotations = slerp(t).as_matrix()
            
            prev_position = interpolated_positions[0]
            prev_rotation = interpolated_rotations[0]
            prev_timestamp = interpolated_timestamps[0]
            
            for j in range(1, len(t)):
                current_position = interpolated_positions[j]
                current_rotation = interpolated_rotations[j]
                current_timestamp = interpolated_timestamps[j]
                
                dt = current_timestamp - prev_timestamp
                
                current_velocity = (current_position - prev_position) / dt
                linear_acceleration = (current_velocity - prev_velocity) / dt
                linear_acceleration = prev_rotation.T @ (linear_acceleration + self.W_gravity)
                
                q_prev = R2quaternion(prev_rotation) # x, y, z, w
                q_current = R2quaternion(current_rotation) # x, y, z, w
                if np.dot(q_prev, q_current) < 0:
                    q_current = -q_current
                    
                q1 = np.array([q_prev[3], q_prev[0], q_prev[1], q_prev[2]]) # w, x, y, z
                q2 = np.array([q_current[3], q_current[0], q_current[1], q_current[2]]) # w, x, y, z
                angular_velocity = (2 / dt) * np.array([q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
                                                        q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
                                                        q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0] ])
                
                
                linear_acceleration_noise = self.accelerometer_noise_density * np.random.normal(0, 1, 3)
                linear_acceleration_bias_noise = self.accelerometer_random_walk * np.random.normal(0, 1, 3)
                gyroscope_noise = self.gyroscope_noise_density * np.random.normal(0, 1, 3)
                gyroscope_bias_noise = self.gyroscope_random_walk * np.random.normal(0, 1, 3)
                          
                accelerometer_bias += linear_acceleration_bias_noise
                linear_acceleration_noisy = linear_acceleration + linear_acceleration_bias_noise + linear_acceleration_noise
                
                gyroscope_bias += gyroscope_bias_noise
                angular_velocity_noisy = angular_velocity + gyroscope_bias_noise + gyroscope_noise
                
                imu_rows_list.append({'timestamp': current_timestamp,
                                        'wx': angular_velocity[0], 'wy': angular_velocity[1], 'wz': angular_velocity[2],
                                        'ax': linear_acceleration[0], 'ay': linear_acceleration[1], 'az': linear_acceleration[2]})
                
                imu_noisy_rows_list.append({'timestamp': current_timestamp,
                                            'wx': angular_velocity_noisy[0], 'wy': angular_velocity_noisy[1], 'wz': angular_velocity_noisy[2],
                                            'ax': linear_acceleration_noisy[0], 'ay': linear_acceleration_noisy[1], 'az': linear_acceleration_noisy[2]})
                
                trajectory_rows_list.append({'timestamp': current_timestamp,
                                            'T00': current_rotation[0, 0], 'T01': current_rotation[0, 1], 'T02': current_rotation[0, 2], 'T03': current_position[0],
                                            'T10': current_rotation[1, 0], 'T11': current_rotation[1, 1], 'T12': current_rotation[1, 2], 'T13': current_position[1],
                                            'T20': current_rotation[2, 0], 'T21': current_rotation[2, 1], 'T22': current_rotation[2, 2], 'T23': current_position[2]})
                                
                prev_velocity = current_velocity
                prev_position = current_position
                prev_rotation = current_rotation
                prev_timestamp = current_timestamp
            
        gt_imu = pd.DataFrame(imu_rows_list)
        gt_imu.to_csv(f'{self.gt_imu_file}', index=False)
        
        gt_trajectory_transforms = pd.DataFrame(trajectory_rows_list)
        gt_trajectory_transforms.to_csv(f'{self.root_folder}/transforms_gt.csv', index=False)
        
        noisy_imu = pd.DataFrame(imu_noisy_rows_list)
        noisy_imu.to_csv(f'{self.noisy_imu_file}', index=False)