import pandas as pd
import numpy as np
import tqdm as tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

BASE_DATASET_PATH = './data'

class Parser():
    def __init__(self, source: str, sequence: str,  gt: bool = True, initial_time_stamp: int = -1, final_time_stamp: int = -1):
        self.source = source
        self.sequence = sequence
        self.root_folder = f'{BASE_DATASET_PATH}/{source}/{sequence}/'
        self.gt = gt
        self.initial_time_stamp = initial_time_stamp
        self.final_time_stamp = final_time_stamp
        
        self.transforms_file = self.root_folder + 'transforms_gt.csv'
        self.imu_file = self.root_folder + 'imu_gt.csv' if gt else self.root_folder + 'imu.csv'
        
        if source == 'synthetic': self.cam_file = self.root_folder + 'camera_gt.csv' if gt else self.root_folder + 'camera.csv'
        else: self.cam_file = self.root_folder + 'camera.csv'
    
    def extract_gt_trajectory(self):
        gt_trajectory = pd.read_csv(self.transforms_file)        
        
        if self.initial_time_stamp == -1 and self.final_time_stamp == -1:
            return gt_trajectory
        elif self.initial_time_stamp == -1 and self.final_time_stamp != -1:
            gt_trajectory = gt_trajectory[gt_trajectory['timestamp'] <= self.final_time_stamp]
            gt_trajectory = gt_trajectory.reset_index(drop=True)  
            return gt_trajectory
        elif self.initial_time_stamp != -1 and self.final_time_stamp == -1:
            gt_trajectory = gt_trajectory[gt_trajectory['timestamp'] >= self.initial_time_stamp]
            gt_trajectory = gt_trajectory.reset_index(drop=True)  
            return gt_trajectory
        else:
            gt_trajectory = gt_trajectory[(gt_trajectory['timestamp'] >= self.initial_time_stamp) & (gt_trajectory['timestamp'] <= self.final_time_stamp)]
            gt_trajectory = gt_trajectory.reset_index(drop=True)
            return gt_trajectory
    
    def extract_imu(self):
        imu = pd.read_csv(self.imu_file)        
        
        if self.initial_time_stamp == -1 and self.final_time_stamp == -1:
            return imu
        elif self.initial_time_stamp == -1 and self.final_time_stamp != -1:
            imu = imu[imu['timestamp'] <= self.final_time_stamp]
            imu = imu.reset_index(drop=True)  
            return imu
        elif self.initial_time_stamp != -1 and self.final_time_stamp == -1:
            imu = imu[imu['timestamp'] >= self.initial_time_stamp]
            imu = imu.reset_index(drop=True)  
            return imu
        else:
            imu = imu[(imu['timestamp'] >= self.initial_time_stamp) & (imu['timestamp'] <= self.final_time_stamp)]
            imu = imu.reset_index(drop=True)
            return imu

    def extract_images(self):
        camera_images = pd.read_csv(self.cam_file)
        
        if self.initial_time_stamp == -1 and self.final_time_stamp == -1:
            return camera_images
        elif self.initial_time_stamp == -1 and self.final_time_stamp != -1:
            camera_images = camera_images[camera_images['timestamp'] <= self.final_time_stamp]
            camera_images = camera_images.reset_index(drop=True)  
            return camera_images
        elif self.initial_time_stamp != -1 and self.final_time_stamp == -1:
            camera_images = camera_images[camera_images['timestamp'] >= self.initial_time_stamp]
            camera_images = camera_images.reset_index(drop=True)  
            return camera_images
        else:
            camera_images = camera_images[(camera_images['timestamp'] >= self.initial_time_stamp) & (camera_images['timestamp'] <= self.final_time_stamp)]
            camera_images = camera_images.reset_index(drop=True)
            return camera_images
        
    def extract_synthetic_camera_measurements(self, image_path: str):
        camera_images = pd.read_csv(image_path)
        
        keypoint_x = camera_images['x'].values
        keypoint_y = camera_images['y'].values
        descriptor_0 = camera_images['descriptor_0'].values
        descriptor_1 = camera_images['descriptor_1'].values
        descriptor_2 = camera_images['descriptor_2'].values
        descriptor_3 = camera_images['descriptor_3'].values
        descriptor_4 = camera_images['descriptor_4'].values
        descriptor_5 = camera_images['descriptor_5'].values
        descriptor_6 = camera_images['descriptor_6'].values
        descriptor_7 = camera_images['descriptor_7'].values
        descriptor_8 = camera_images['descriptor_8'].values
        descriptor_9 = camera_images['descriptor_9'].values
        scores = camera_images['score'].values
        
        keypoints = np.vstack((keypoint_x, keypoint_y)).T
        descriptors = np.vstack((descriptor_0, descriptor_1, descriptor_2, descriptor_3, descriptor_4, descriptor_5, descriptor_6, descriptor_7, descriptor_8, descriptor_9)).T
        scores = np.array(scores)
        
        return keypoints, descriptors, scores
    
    def extract_gt_world_points(self):
        file = self.root_folder + 'world_points.csv'
        world_points = pd.read_csv(file)
        world_points = world_points[['x', 'y', 'z']].values
        return world_points