import sys
sys.path.append('../')

from dataclasses import dataclass, field
from typing import List

import os
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Slerp

from src.msckf.Camera import Camera
from src.utils.geometry import *
from src.utils.visualization_utils import *

def format_nparray(nparray: np.ndarray) -> str:
    return ','.join(f"{x:.6f}" for x in nparray)

@dataclass
class CameraParameters:
    K: np.ndarray = np.array([[180,   0, 320], [  0, 180, 240], [  0,   0,   1]])
    T_W_C: Isometry3D = Isometry3D(np.eye(3), np.zeros(3))
    width: int = 640
    height: int = 480
    sigma_noise: float = 0

@dataclass
class CameraMeasurement:
    positions: List[np.ndarray] = field(default_factory=list)
    descriptors: List[np.ndarray] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)
    
@dataclass
class IMUParameters:
    sigma_noise_angular_velocity: float = 0
    sigma_noise_linear_acceleration: float = 0
    sigma_noise_bias_gyroscope: float = 0
    sigma_noise_bias_accelerometer: float = 0
    T_W_I: Isometry3D = Isometry3D(np.eye(3), np.zeros(3))
    
@dataclass
class IMUMeasurement:
    angular_velocity: np.ndarray = np.zeros((0, 3))
    linear_acceleration: np.ndarray = np.zeros((0, 3))

@dataclass
class WorldPoints:
    positions: List[np.ndarray] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)
    descriptors: List[np.ndarray] = field(default_factory=list)
    
@dataclass
class Frame:
    timestamp: float = 0
    pose: Isometry3D = Isometry3D(np.eye(3), np.zeros(3))
    IMU_measurement_gt: IMUMeasurement = IMUMeasurement()
    IMU_measurement_noisy: IMUMeasurement = IMUMeasurement()
    camera_measurement_gt: CameraMeasurement = CameraMeasurement()
    camera_measurement_noisy: CameraMeasurement = CameraMeasurement()

class TrajectorySegment():
    def __init__(self, position_waypoints: np.ndarray = np.zeros(3), orientation_waypoints: np.ndarray = np.zeros(3)) -> None:
        
        self.position_waypoints = position_waypoints
        self.orientations_waypoints = orientation_waypoints 
        if len(self.position_waypoints) > 3:
            raise ValueError('Define a segment with 2 waypoints to be linear or 3 waypoints to be cubic.')
    
        self.type = 'linear' if len(self.position_waypoints) == 2 else 'cubic'
        self.poses: List[Isometry3D] = []
        
    def generate(self, rate: float) -> np.ndarray:      
              
        samples_per_meter = rate
        
        if self.type == 'linear':
            distance = np.linalg.norm(np.array(self.position_waypoints[1]) - np.array(self.position_waypoints[0]))
            n_samples = int(distance * samples_per_meter)
                        
            x = np.linspace(self.position_waypoints[0][0], self.position_waypoints[1][0], n_samples)
            y = np.linspace(self.position_waypoints[0][1], self.position_waypoints[1][1], n_samples)
            z = np.linspace(self.position_waypoints[0][2], self.position_waypoints[1][2], n_samples)
                        
            orientation_1 = euler2R(self.orientations_waypoints[0])
            orientation_2 = euler2R(self.orientations_waypoints[1])
            t = np.linspace(0, 1, n_samples)
            slerp = Slerp([0, 1], scipyR.from_matrix([orientation_1, orientation_2]))
            interpolated_rotations = slerp(t).as_matrix()
            
            for i in range(n_samples):
                self.poses.append(Isometry3D(interpolated_rotations[i], np.array([x[i], y[i], z[i]])))
            
        elif self.type == 'cubic':
            distance = np.linalg.norm(np.array(self.position_waypoints[1]) - np.array(self.position_waypoints[0])) + np.linalg.norm(np.array(self.position_waypoints[2]) - np.array(self.position_waypoints[1]))
            n_samples = int(distance * samples_per_meter)
                        
            x = self.position_waypoints[:, 0]
            y = self.position_waypoints[:, 1]
            z = self.position_waypoints[:, 2]
            
            t = np.linspace(0, 1, len(x))
            cs_x = CubicSpline(t, x)
            cs_y = CubicSpline(t, y)
            cs_z = CubicSpline(t, z)
            
            t_new = np.linspace(0, 1, n_samples)
            x_new = cs_x(t_new)
            y_new = cs_y(t_new)
            z_new = cs_z(t_new)
            
            orientation_1 = euler2R(self.orientations_waypoints[0])
            orientation_2 = euler2R(self.orientations_waypoints[1])
            slerp = Slerp([0, 1], scipyR.from_matrix([orientation_1, orientation_2]))
            interpolated_rotations = slerp(t_new).as_matrix()
            
            for i in range(n_samples):
                self.poses.append(Isometry3D(interpolated_rotations[i], np.array([x_new[i], y_new[i], z_new[i]])))
            
        return self.poses
    
class MeasurementsGenerator():
    def __init__(self, segments: List[TrajectorySegment], IMU_parameters: IMUParameters, camera_parameters: CameraParameters, world_points: WorldPoints, rate: float) -> None:
        self.segments: List[TrajectorySegment] = segments
        self.imu_parameters: IMUParameters = IMU_parameters
        self.camera_parameters: CameraParameters = camera_parameters
        self.world_points: WorldPoints = world_points
        self.frames: List[Frame] = []
        self.rate = rate
        self.delta_t = 1/rate
    
    def generate(self):
        
        #* Generate poses
        poses = [pose for segment in self.segments for pose in segment.generate(rate=self.rate)]
        
        #*Stationary init with 19 frames
        poses = [Isometry3D(np.eye(3), np.zeros(3))]*19 + poses
    
        
        #* Generate IMU measurements
        W_gravity = np.array([0, 0, -9.81])
        gyroscope_bias = np.zeros(3)
        accelerometer_bias = np.zeros(3)
        
        imu_measurements_gt = []
        imu_measurements_noisy = []
        prev_velocity = np.zeros(3)
        for i in range(1, len(poses)):
            
            prev_position = poses[i-1].t
            prev_rotation = poses[i-1].R
            position = poses[i].t
            rotation = poses[i].R
            
            velocity = (position - prev_position) / self.delta_t
            linear_acceleration = prev_rotation.T @ ( ((velocity - prev_velocity) / self.delta_t) + W_gravity )

            q_prev = R2quaternion(prev_rotation)
            q_current = R2quaternion(rotation)
            q1 = np.array([q_prev[3], q_prev[0], q_prev[1], q_prev[2]])
            q2 = np.array([q_current[3], q_current[0], q_current[1], q_current[2]])
            angular_velocity = (2 / self.delta_t) * np.array([ q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
                                                               q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
                                                               q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0] ])
            
            imu_measurement_gt = IMUMeasurement(angular_velocity=angular_velocity, linear_acceleration=linear_acceleration)
            imu_measurements_gt.append(imu_measurement_gt)
            
            noise_angular_velocity, _ = white_gaussian_noise([self.imu_parameters.sigma_noise_angular_velocity, self.imu_parameters.sigma_noise_angular_velocity, self.imu_parameters.sigma_noise_angular_velocity])
            angular_velocity += noise_angular_velocity
            noise_linear_acceleration, _ = white_gaussian_noise([self.imu_parameters.sigma_noise_linear_acceleration, self.imu_parameters.sigma_noise_linear_acceleration, self.imu_parameters.sigma_noise_linear_acceleration])
            linear_acceleration += noise_linear_acceleration
            
            gyroscope_bias_noise, _ = white_gaussian_noise([self.imu_parameters.sigma_noise_bias_gyroscope, self.imu_parameters.sigma_noise_bias_gyroscope, self.imu_parameters.sigma_noise_bias_gyroscope])
            gyroscope_bias += gyroscope_bias_noise
            angular_velocity += gyroscope_bias
            
            accelerometer_bias_noise, _ = white_gaussian_noise([self.imu_parameters.sigma_noise_bias_accelerometer, self.imu_parameters.sigma_noise_bias_accelerometer, self.imu_parameters.sigma_noise_bias_accelerometer])
            accelerometer_bias += accelerometer_bias_noise
            linear_acceleration += accelerometer_bias
            
            imu_measurement_noisy = IMUMeasurement(angular_velocity=angular_velocity, linear_acceleration=linear_acceleration)
            imu_measurements_noisy.append(imu_measurement_noisy)
            
            prev_velocity = velocity
            
        #* Generate camera measurements
        camera_measurements_gt = []
        camera_measurements_noisy = []
        for i in range(len(poses)):       
            t_W_Ii = poses[i].t
            R_W_Ii = poses[i].R
                
            T_W_Ii = Isometry3D(R_W_Ii, t_W_Ii)
            T_I_C = self.imu_parameters.T_W_I.inv() * self.camera_parameters.T_W_C
            T_W_Ci = T_W_Ii * T_I_C
            
            camera = Camera(self.camera_parameters.K, self.camera_parameters.width, self.camera_parameters.height, T_W_Ci)
            camera_measurement_gt = CameraMeasurement(positions=[], descriptors=[], scores=[], indices=[])
            camera_measurement_noisy = CameraMeasurement(positions=[], descriptors=[], scores=[], indices=[])
            for i in range(len(self.world_points.positions)):
                id = self.world_points.indices[i]
                descriptor = self.world_points.descriptors[i]
                W_p = self.world_points.positions[i]
                Ci_p = camera.W2Ci(W_p)
                res, Im_p = camera.project_point(Ci_p)
                if res:
                    camera_measurement_gt.positions.append(Im_p)
                    camera_measurement_gt.descriptors.append(descriptor)
                    camera_measurement_gt.indices.append(id)
                    camera_measurement_gt.scores.append(1)
                    
                    noise, cov = white_gaussian_noise([self.camera_parameters.sigma_noise, self.camera_parameters.sigma_noise])
                    Im_p += noise
                    camera_measurement_noisy.positions.append(Im_p)
                    camera_measurement_noisy.descriptors.append(descriptor)
                    camera_measurement_noisy.scores.append(1/(1+np.trace(cov)))
                    camera_measurement_noisy.indices.append(id)
                    
            camera_measurements_gt.append(camera_measurement_gt)
            camera_measurements_noisy.append(camera_measurement_noisy)
        
        for i in range(len(imu_measurements_noisy)):
            self.frames.append(Frame(timestamp=i*self.delta_t, pose=poses[i], 
                                     IMU_measurement_gt=imu_measurements_gt[i], IMU_measurement_noisy=imu_measurements_noisy[i], 
                                     camera_measurement_gt=camera_measurements_gt[i], camera_measurement_noisy=camera_measurements_noisy[i]))        

    def save(self, folder_path: str) -> None:
        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(f'{folder_path}/cam_gt', exist_ok=True)
        os.makedirs(f'{folder_path}/cam_noisy', exist_ok=True)
        
        transforms_file = f'{folder_path}/transforms_gt.csv'
        gt_imu_file = f'{folder_path}/imu_gt.csv'
        noisy_imu_file = f'{folder_path}/imu.csv' 
        camera_file_gt = f'{folder_path}/camera_gt.csv'
        camera_file_noisy = f'{folder_path}/camera.csv'
        
        transforms_rows = []
        imu_gt_rows = []
        imu_noisy_rows = []
        camera_gt_rows = []
        camera_noisy_rows = []
        camera_measurements_gt_rows = []
        camera_measurements_noisy_rows = []
        camera_indices = []
        for i in range(len(self.frames)):
            frame = self.frames[i]
            timestamp = frame.timestamp
            
            pose = frame.pose
            t_W_C = pose.t
            R_W_C = pose.R
            transforms_rows.append({'timestamp': timestamp, 
                                    'T00': R_W_C[0, 0], 'T01': R_W_C[0, 1], 'T02': R_W_C[0, 2], 'T03': t_W_C[0],
                                    'T10': R_W_C[1, 0], 'T11': R_W_C[1, 1], 'T12': R_W_C[1, 2], 'T13': t_W_C[1],
                                    'T20': R_W_C[2, 0], 'T21': R_W_C[2, 1], 'T22': R_W_C[2, 2], 'T23': t_W_C[2]})
            
            imu_measurement_gt = frame.IMU_measurement_gt
            angular_velocity_gt = imu_measurement_gt.angular_velocity
            linear_acceleration_gt = imu_measurement_gt.linear_acceleration
            imu_gt_rows.append({'timestamp': timestamp,
                                'wx': angular_velocity_gt[0], 'wy': angular_velocity_gt[1], 'wz': angular_velocity_gt[2],
                                'ax': linear_acceleration_gt[0], 'ay': linear_acceleration_gt[1], 'az': linear_acceleration_gt[2]})
            
            imu_measurement_noisy = frame.IMU_measurement_noisy
            angular_velocity_noisy = imu_measurement_noisy.angular_velocity
            linear_acceleration_noisy = imu_measurement_noisy.linear_acceleration
            imu_noisy_rows.append({'timestamp': timestamp,
                                   'wx': angular_velocity_noisy[0], 'wy': angular_velocity_noisy[1], 'wz': angular_velocity_noisy[2],
                                   'ax': linear_acceleration_noisy[0], 'ay': linear_acceleration_noisy[1], 'az': linear_acceleration_noisy[2]})
            
            if i%10 != 0: continue
            camera_indices.append(i)
            camera_gt_rows.append({'timestamp': timestamp, 'image_path': f'{folder_path}/cam_gt/{i:05d}.csv'})
            camera_noisy_rows.append({'timestamp': timestamp, 'image_path': f'{folder_path}/cam_noisy/{i:05d}.csv'})
            
            camera_measurement_gt_rows = []
            camera_measurement_noisy_rows = []
            for j in range(len(frame.camera_measurement_gt.positions)):
                position = frame.camera_measurement_gt.positions[j]
                descriptor = frame.camera_measurement_gt.descriptors[j]
                score = frame.camera_measurement_gt.scores[j]
                id = frame.camera_measurement_gt.indices[j]
                camera_measurement_gt_rows.append({'timestamp': timestamp, 'id': id, 'x': position[0], 'y': position[1], 'score': score, 
                                                   'descriptor_0': descriptor[0], 'descriptor_1': descriptor[1], 'descriptor_2': descriptor[2], 'descriptor_3': descriptor[3], 'descriptor_4': descriptor[4], 
                                                   'descriptor_5': descriptor[5], 'descriptor_6': descriptor[6], 'descriptor_7': descriptor[7], 'descriptor_8': descriptor[8], 'descriptor_9': descriptor[9]})
                
                position = frame.camera_measurement_noisy.positions[j]
                descriptor = frame.camera_measurement_noisy.descriptors[j]
                score = frame.camera_measurement_noisy.scores[j]
                id = frame.camera_measurement_noisy.indices[j]
                camera_measurement_noisy_rows.append({'timestamp': timestamp, 'id': id, 'x': position[0], 'y': position[1], 'score': score, 
                                                      'descriptor_0': descriptor[0], 'descriptor_1': descriptor[1], 'descriptor_2': descriptor[2], 'descriptor_3': descriptor[3], 'descriptor_4': descriptor[4],
                                                      'descriptor_5': descriptor[5], 'descriptor_6': descriptor[6], 'descriptor_7': descriptor[7], 'descriptor_8': descriptor[8], 'descriptor_9': descriptor[9]})
            
            camera_measurements_gt_rows.append(camera_measurement_gt_rows)
            camera_measurements_noisy_rows.append(camera_measurement_noisy_rows)
            
        gt_imu = pd.DataFrame(imu_gt_rows)
        gt_imu.to_csv(gt_imu_file, index=False)
        
        noisy_imu = pd.DataFrame(imu_noisy_rows)
        noisy_imu.to_csv(noisy_imu_file, index=False)
        
        transforms = pd.DataFrame(transforms_rows)
        transforms.to_csv(transforms_file, index=False)
        
        camera_gt = pd.DataFrame(camera_gt_rows)
        camera_gt.to_csv(camera_file_gt, index=False)
        
        camera_noisy = pd.DataFrame(camera_noisy_rows)
        camera_noisy.to_csv(camera_file_noisy, index=False)
        
        for i in range(len(camera_measurements_gt_rows)):
            index = camera_indices[i]
            camera_measurements_gt = pd.DataFrame(camera_measurements_gt_rows[i])
            camera_measurements_gt.to_csv(f'{folder_path}/cam_gt/{index:05d}.csv', index=False)
            
            camera_measurements_noisy = pd.DataFrame(camera_measurements_noisy_rows[i])
            camera_measurements_noisy.to_csv(f'{folder_path}/cam_noisy/{index:05d}.csv', index=False)
        

class WorldPointsGenerator():
    def __init__(self) -> None:
        self.last_world_point_id = 0
        self.world_points: WorldPoints = WorldPoints()
        
    def add_random_world_points(self, n_points: int, scale_x: float = 1, scale_y: float = 1, scale_z: float = 1, origin: np.ndarray = np.array([0, 0, 0])) -> None:        
        points = np.random.rand(n_points, 3)
        points[:, 0] *= scale_x
        points[:, 1] *= scale_y
        points[:, 2] *= scale_z
        
        points += origin
        
        for point in points:
            self.world_points.positions.append(point)
            self.world_points.indices.append(self.last_world_point_id)
            descriptor = np.random.rand(10)
            descriptor = descriptor / np.linalg.norm(descriptor)
            self.world_points.descriptors.append(descriptor)
            self.last_world_point_id += 1

    def save(self, folder_path: str) -> None:
        world_point_rows = []
        for i in range(len(self.world_points.positions)):
            position = self.world_points.positions[i]
            id = self.world_points.indices[i]
            descriptor = self.world_points.descriptors[i]
            world_point_rows.append({'id': id, 'x': position[0], 'y': position[1], 'z': position[2],
                                     'descriptor_0': descriptor[0], 'descriptor_1': descriptor[1], 'descriptor_2': descriptor[2], 'descriptor_3': descriptor[3], 'descriptor_4': descriptor[4],
                                     'descriptor_5': descriptor[5], 'descriptor_6': descriptor[6], 'descriptor_7': descriptor[7], 'descriptor_8': descriptor[8], 'descriptor_9': descriptor[9]})
        
        world_points = pd.DataFrame(world_point_rows)
        world_points.to_csv(f'{folder_path}/world_points.csv', index=False)
        

#| --------------------------------------------------------------------------------------------------------------------------------------- |#
#* Synthetic dataset generation

camera_parameters = CameraParameters(K=np.array([[180,   0, 320], [  0, 180, 240], [  0,   0,   1]]), 
                                     T_W_C=Isometry3D(np.array([[ 0,  0, 1], [-1,  0, 0], [ 0, -1, 0]]), np.array([0, 0, 0])), 
                                     width=640, 
                                     height=480, 
                                     sigma_noise=0.01)

imu_parameters = IMUParameters(T_W_I=Isometry3D(np.eye(3), np.zeros(3)), 
                               sigma_noise_linear_acceleration=0.0001, 
                               sigma_noise_angular_velocity=0.00001, 
                               sigma_noise_bias_accelerometer=0.00001,
                               sigma_noise_bias_gyroscope=0.000001)

dt = 0.005
rate = 1/dt
save_folder = "/home/valeriospagnoli/Thesis/vio/dataset/synthetic/classic"

if os.path.exists(save_folder): 
    input("The folder already exists. Press Enter to delete it and create a new one.")
    os.system(f"rm -r {save_folder}")
os.makedirs(save_folder, exist_ok=True)

#* Circular
world_points_generator = WorldPointsGenerator()
world_points_generator.add_random_world_points(400, 12, 12, 5, np.array([-6, -4, 0]))
world_points = world_points_generator.world_points

segment_1 = TrajectorySegment(position_waypoints=np.array([[0,0,0], [np.sqrt(2),2-np.sqrt(2),0], [2,2,0]]), 
                              orientation_waypoints=np.array([[0, 0, 0], [0, 0, np.pi/2]]))
segment_2 = TrajectorySegment(position_waypoints=np.array([[2,2,0], [np.sqrt(2),2+np.sqrt(2),0], [0,4,0]]),
                              orientation_waypoints=np.array([[0, 0, np.pi/2], [0, 0, np.pi]]))
segment_3 = TrajectorySegment(position_waypoints=np.array([[0,4,0], [-np.sqrt(2),2+np.sqrt(2),0], [-2,2,0]]),
                              orientation_waypoints=np.array([[0, 0, np.pi], [0, 0, 3*np.pi/2]]))
segment_4 = TrajectorySegment(position_waypoints=np.array([[-2,2,0], [-np.sqrt(2),2-np.sqrt(2),0], [0,0,0]]),
                              orientation_waypoints=np.array([[0, 0, 3*np.pi/2], [0, 0, 0]]))

segments = [segment_1, segment_2, segment_3, segment_4]

#* Classic
# world_points_generator = WorldPointsGenerator()
# world_points_generator.add_random_world_points(200, 35, 35, 5, np.array([-10, -10, 0]))
# world_points = world_points_generator.world_points

# segment_1 = TrajectorySegment(position_waypoints=np.array([[0,0,0], [10,0,0]]), 
#                               orientation_waypoints=np.array([[0, 0, 0], [0, 0, np.pi/2]]))
# segment_2 = TrajectorySegment(position_waypoints=np.array([[10,0,0], [13.8,1.2,0], [15, 5, 0]]),
#                               orientation_waypoints=np.array([[0, 0, np.pi/2], [0, 0, np.pi/2]]))
# segment_3 = TrajectorySegment(position_waypoints=np.array([[15,5,0], [15,10,0]]),
#                               orientation_waypoints=np.array([[0, 0, np.pi/2], [0, 0, np.pi]]))
# segment_4 = TrajectorySegment(position_waypoints=np.array([[15,10,0], [13.8,13.8,0], [10,15,0]]),
#                               orientation_waypoints=np.array([[0, 0, np.pi], [0, 0, np.pi]]))
# segment_5 = TrajectorySegment(position_waypoints=np.array([[10,15,0], [5,15,0]]),
#                               orientation_waypoints=np.array([[0, 0, np.pi], [0, 0, 3*np.pi/2]]))
# segment_6 = TrajectorySegment(position_waypoints=np.array([[5,15,0], [1.2,13.8,0], [0, 10, 0]]),
#                               orientation_waypoints=np.array([[0, 0, 3*np.pi/2], [0, 0, 3*np.pi/2]]))
# segment_7 = TrajectorySegment(position_waypoints=np.array([[0,10,0], [0,0,0]]),
#                               orientation_waypoints=np.array([[0, 0, 3*np.pi/2], [0, 0, 2*np.pi]]))

# segments = [segment_1, segment_2, segment_3, segment_4, segment_5, segment_6, segment_7]

measurement_generator = MeasurementsGenerator(segments=segments, 
                                              IMU_parameters=imu_parameters, camera_parameters=camera_parameters, 
                                              world_points=world_points, rate=rate)
measurement_generator.generate()
world_points_generator.save(save_folder)
measurement_generator.save(save_folder)

print("Number of frames: ", len(measurement_generator.frames))  
print("Number of world points: ", len(world_points.positions))  

canvas_3D = Canvas3D(x_range=[-15, 30], y_range=[-15, 30], z_range=[-1, 5])
canvas_3D.add_points(np.array(world_points.positions), color='blue', size=1)
canvas_3D.add_points(np.array([frame.pose.t for frame in measurement_generator.frames]), color='red', size=1)
canvas_3D.show()
