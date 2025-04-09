import sys
sys.path.append('../../')

import numpy as np
import pandas as pd
import rerun as rr
import cv2
from collections import deque
from scipy.stats import chi2
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from src.utils.geometry import *
from src.utils.visualization_utils import *
from src.msckf.IMU import *
from src.msckf.MSCKF import *
from src.msckf.Camera import *
from dataset.tools import parser
from dataset.tools.dataset_generators.photorealistic_generator import PhotorealisticGenerator

np.random.seed(42)

BASE_DATASET_PATH = '/home/valeriospagnoli/Thesis/vio/dataset'
source = 'tartanair' # ['synthetic', 'peringlab', 'tartanair']
sequence = 'ME007'  
max_frames = 2500
noise_level = 'mid' # ['low', 'mid', 'high']

print(f'Running VIO on {source} dataset, sequence {sequence} with {noise_level} noise level')

camera_info = pd.read_csv(f'{BASE_DATASET_PATH}/{source}/camera_info.csv')
T_W_C = Isometry3D(np.array([[0,0,1], [-1,0,0], [0,-1,0]]),  np.zeros(3))
K = np.array([[camera_info.iloc[0]['fx'], 0, camera_info.iloc[0]['px']],
              [0, camera_info.iloc[0]['fy'], camera_info.iloc[0]['py']],
              [0, 0, 1]])

width = camera_info.iloc[0]['w']
height = camera_info.iloc[0]['h']

save_results = False
experiment_folder = f'/home/valeriospagnoli/Thesis/vio/experiments/{source}/{sequence}/{noise_level}_noise'


#** Rerun setup
# ========================================================================================== #

log_images = False
# rr.init('vio')
# rr.spawn()
if save_results: rr.save(f'{experiment_folder}/recording.rrd')

rr.log('relative_translation_error/t', rr.SeriesLine(name='Relative Translation Error VIO', color=[0, 0, 255]), static=True)
rr.log('relative_orientation_error/r', rr.SeriesLine(name='Relative Orientation Error VIO', color=[0, 0, 255]), static=True)

rr.log('absolute_translation_error/x', rr.SeriesLine(name='Absolute Translation Error x', color=[255, 0, 0]), static=True)
rr.log('absolute_translation_error/xlb', rr.SeriesLine(name='lower bound', color=[200, 150, 0]), static=True)
rr.log('absolute_translation_error/xub', rr.SeriesLine(name='upper bound', color=[200, 150, 0]), static=True)

rr.log('absolute_translation_error/y', rr.SeriesLine(name='Absolute Translation Error y', color=[0, 255, 0]), static=True)
rr.log('absolute_translation_error/ylb', rr.SeriesLine(name='lower bound', color=[200, 150, 0]), static=True)
rr.log('absolute_translation_error/yub', rr.SeriesLine(name='upper bound', color=[200, 150, 0]), static=True)

rr.log('absolute_translation_error/z', rr.SeriesLine(name='Absolute Translation Error z', color=[0, 0, 255]), static=True)
rr.log('absolute_translation_error/zlb', rr.SeriesLine(name='lower bound', color=[200, 150, 0]), static=True)
rr.log('absolute_translation_error/zub', rr.SeriesLine(name='lower bound', color=[200, 150, 0]), static=True)

rr.log('absolute_orientation_error/roll', rr.SeriesLine(name='Absolute Orientation Error roll', color=[255, 0, 0]), static=True)
rr.log('absolute_orientation_error/rolllb', rr.SeriesLine(name='lower bound', color=[200, 150, 0]), static=True)
rr.log('absolute_orientation_error/rollub', rr.SeriesLine(name='upper bound', color=[200, 150, 0]), static=True)

rr.log('absolute_orientation_error/pitch', rr.SeriesLine(name='Absolute Orientation Error pitch', color=[0, 255, 0]), static=True)
rr.log('absolute_orientation_error/pitchlb', rr.SeriesLine(name='lower bound', color=[200, 150, 0]), static=True)
rr.log('absolute_orientation_error/pitchub', rr.SeriesLine(name='upper bound', color=[200, 150, 0]), static=True)

rr.log('absolute_orientation_error/yaw', rr.SeriesLine(name='Absolute Orientation Error yaw', color=[0, 0, 255]), static=True)
rr.log('absolute_orientation_error/yawlb', rr.SeriesLine(name='lower bound', color=[200, 150, 0]), static=True)
rr.log('absolute_orientation_error/yawub', rr.SeriesLine(name='upper bound', color=[200, 150, 0]), static=True)

rr.log('msckf/features', rr.SeriesLine(name='Number of Features', color=[200, 150, 0]), static=True)
rr.log('msckf/camera_states', rr.SeriesLine(name='Number of Camera States', color=[200, 150, 0]), static=True)
rr.log('msckf/NEES/metric', rr.SeriesLine(name='NEES', color=[200, 150, 0]), static=True)
rr.log('msckf/NEES/compare_lower', rr.SeriesLine(name='Chi Square(0.05/2, 6)', color=[200, 0, 0]), static=True)
rr.log('msckf/NEES/compare_upper', rr.SeriesLine(name='Chi Square(1 - 0.05/2, 6)', color=[0, 200, 0]), static=True)

rr_trajectory_radii = 0.01
rr_point_radii = 0.01
rr_axis_length = 0.2


#** Noise parameters
# ========================================================================================== #

accelerometer_noise_densities = [0.01, 0.005, 0.001]
gyroscope_noise_densities = [0.001, 0.0005, 0.0001]
accelerometer_random_walks = [0.001, 0.0005, 0.0001]
gyroscope_random_walks = [0.0001, 0.00005, 0.00001]

if noise_level == 'high':
    accelerometer_noise_density = accelerometer_noise_densities[0]
    gyroscope_noise_density = gyroscope_noise_densities[0]
    accelerometer_random_walk = accelerometer_random_walks[0]
    gyroscope_random_walk = gyroscope_random_walks[0]
    
elif noise_level == 'mid':
    accelerometer_noise_density = accelerometer_noise_densities[1]
    gyroscope_noise_density = gyroscope_noise_densities[1]
    accelerometer_random_walk = accelerometer_random_walks[1]
    gyroscope_random_walk = gyroscope_random_walks[1]
    
elif noise_level == 'low':
    accelerometer_noise_density = accelerometer_noise_densities[2]
    gyroscope_noise_density = gyroscope_noise_densities[2]
    accelerometer_random_walk = accelerometer_random_walks[2]
    gyroscope_random_walk = gyroscope_random_walks[2]

#** Photorealistic generator
# ========================================================================================== #
if source != 'synthetic':
    generator = PhotorealisticGenerator(source=source, sequence=sequence,
                                        accelerometer_noise_density=accelerometer_noise_density,
                                        gyroscope_noise_density=gyroscope_noise_density,
                                        accelerometer_random_walk=accelerometer_random_walk,
                                        gyroscope_random_walk=gyroscope_random_walk)
    generator.process_data()

#** Parsing
# ========================================================================================== #
data_parser = parser.Parser(source=source, sequence=sequence, gt=False, initial_time_stamp=-1, final_time_stamp=-1)
gt_trajectory = data_parser.extract_gt_trajectory()
imu_data = data_parser.extract_imu()
camera_images = data_parser.extract_images()


#** MSCKF
# ========================================================================================== #
msckf_parameters = MSCKFParameters(
    #* Camera parameters
    T_W_C = T_W_C,
    K = K,
    width = width,
    height = height,
    sigma_image = 0.1,
    
    #* IMU parameters
    only_imu = False,
    accelerometer_noise_density=accelerometer_noise_density,
    gyroscope_noise_density=gyroscope_noise_density,
    accelerometer_random_walk=accelerometer_random_walk,
    gyroscope_random_walk=gyroscope_random_walk,
    W_gravity=np.array([0, 0, -9.81]),
    
    #* Feature parameters
    number_of_extracted_features=300,
    min_cosine_similarity=0.95,
    use_parallax=True,
    min_parallax=45,
    epipolar_rejection_threshold=0.005,
    homography_rejection_threshold=5,
    min_number_of_frames_to_be_tracked=4,
    min_number_of_frames_to_be_lost=2,
    max_number_of_camera_states=30
)

msckf = MSCKF(parameters=msckf_parameters, rr=None)


#** Metrics
# ========================================================================================== #

#* Trajectories
gt_positions = []
gt_rotations = []
estimated_positions = []
estimated_rotations = []

#* Relative Translation Error (RTE) and Relative Orientation Error (ROE)
rte_array = []
roe_array = []

#* Absolute Translation Error (ATE) and Absolute Orientation Error (AOE)
ate_x = []
ate_x_bounds = []
ate_y = []
ate_y_bounds = []
ate_z = []
ate_z_bounds = []
aoe_roll = []
aoe_roll_bounds = []
aoe_pitch = []
aoe_pitch_bounds = []
aoe_yaw = []
aoe_yaw_bounds = []

#* Relative Root Mean Squared Error (RMSE)
relative_RMSE_position = []
relative_RMSE_orientation = [] 

#* Profiling
loop_time = []
loop_time_with_camera = []
loop_time_without_camera = []


#** Initialize the test
# ========================================================================================== #
last_camera_time_index = 1
T_W_I0_gt = Isometry3D(np.eye(3), np.zeros(3))
T_W_I0_est = Isometry3D(np.eye(3), np.zeros(3))

relative_position_error_deque = deque(maxlen=10)
relative_orientation_error_deque = deque(maxlen=10)

#** Loop
# ========================================================================================== #
for i in tqdm(range(max_frames)):     
    if i > len(imu_data)-1: break
    if last_camera_time_index+1 > len(camera_images)-1: break
    
    start_loop = time.time()
    
    #** Get measurements
    # .......................................................................................... #
    imu_timestamp = imu_data.iloc[i]['timestamp']
    camera_timestamp = camera_images.iloc[last_camera_time_index]['timestamp']
    angular_velocity = np.array([imu_data.iloc[i]['wx'], imu_data.iloc[i]['wy'], imu_data.iloc[i]['wz']])
    linear_acceleration = np.array([imu_data.iloc[i]['ax'], imu_data.iloc[i]['ay'], imu_data.iloc[i]['az']])
    imu_measurement = IMUMeasurement(imu_timestamp, angular_velocity, linear_acceleration)

    #** Test VIO with MSCKF
    # .......................................................................................... #
    is_there_camera_measurement = False
    msckf.imu_callback(imu_measurement)
    if np.abs(np.round(imu_timestamp - camera_timestamp, 3)) < 0.00001:
        is_there_camera_measurement = True
        image_path = camera_images.iloc[last_camera_time_index]['image_path']
        last_camera_time_index += 1
        
        if source == 'synthetic':
            keypoints, descriptors, scores = data_parser.extract_synthetic_camera_measurements(image_path)
            camera_measurement = CameraMeasurement(keypoints=keypoints, descriptors=descriptors, scores=scores)
            
            image = np.ones((480, 640, 3), np.uint8) * 255
            for p in range(0, 640, 20): cv2.line(image, (p, 0), (p, 480), (200, 200, 200), 1)
            for p in range(0, 480, 20): cv2.line(image, (0, p), (640, p), (200, 200, 200), 1)
            cv2.rectangle(image, (0, 0), (640, 480), (0, 0, 0), 2)
            
            for keypoint in keypoints: cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), 3, (0, 0, 255), -1)
            msckf.feature_callback(image, camera_measurement)
        
        else:
            image = cv2.imread(image_path)     
            msckf.feature_callback(image)     
        
        if log_images: 
            rerun_image = msckf.composed_image if msckf.composed_image is not None else msckf.current_image
            compressed_image = cv2.resize(rerun_image, (640*2, 480), interpolation=cv2.INTER_AREA)
            rr.log('/camera_image', rr.Image(compressed_image, color_model='bgr'))      
    
    time.sleep(0.005)
    end_loop = time.time()
    delta_time = end_loop - start_loop
    loop_time.append(delta_time)
    if is_there_camera_measurement: loop_time_with_camera.append(delta_time)
    else: loop_time_without_camera.append(delta_time)
    
    #** Errors
    # .......................................................................................... #
                
    #* Ground truth
    gt_transform = gt_trajectory.iloc[i]
    gt_position = np.array([gt_transform['T03'], gt_transform['T13'], gt_transform['T23']])
    gt_rotation = np.array([[gt_transform['T00'], gt_transform['T01'], gt_transform['T02']],
                            [gt_transform['T10'], gt_transform['T11'], gt_transform['T12']],
                            [gt_transform['T20'], gt_transform['T21'], gt_transform['T22']]])
    T_W_I1_gt = Isometry3D(gt_rotation, gt_position)
    T_I0_I1_gt = T_W_I0_gt.inv() * T_W_I1_gt
    
    relative_translation_gt = np.linalg.norm(T_I0_I1_gt.t)
    relative_orientation_gt = np.arccos(np.clip((np.trace(T_I0_I1_gt.R) - 1) / 2.0, -1.0, 1.0))
    
    #* Estimate VIO
    T_W_I1_est = msckf.state.imu.T_W_Ii
    T_I0_I1_est = T_W_I0_est.inv() * T_W_I1_est
    
    #* Absolute Pose Error
    abs_T_error = T_W_I1_gt.inv() * T_W_I1_est

    IMU_position_covariance = msckf.state.covariance[12:15, 12:15]
    IMU_orientation_covariance = msckf.state.covariance[:3, :3]
    
    sigmas_position = np.sqrt(np.diag(IMU_position_covariance)) 
    sigmas_orientation = np.sqrt(np.diag(IMU_orientation_covariance))
    x_bound = 3*sigmas_position[0]
    y_bound = 3*sigmas_position[1]
    z_bound = 3*sigmas_position[2]
    roll_bound = 3*sigmas_orientation[0]
    pitch_bound = 3*sigmas_orientation[1]
    yaw_bound = 3*sigmas_orientation[2]
    
    #* Relative Pose Error
    rel_T_error = T_I0_I1_gt.inv() * T_I0_I1_est
    relative_translation_error = np.linalg.norm(rel_T_error.t)
    relative_orientation_error = np.arccos(np.clip((np.trace(rel_T_error.R) - 1) / 2.0, -1.0, 1.0))
    
    rte = (relative_translation_error / relative_translation_gt) if relative_translation_gt != 0 else (1 if relative_translation_gt > 0 else 0)
    roe = (relative_orientation_error / relative_orientation_gt) if relative_orientation_gt != 0 else (1 if relative_orientation_gt > 0 else 0)
    
    if len(relative_position_error_deque)>0 and rte > 10*np.mean(relative_position_error_deque):
        rte_smoothed = 0.001 * rte + 0.999 * np.mean(relative_position_error_deque)
    else: rte_smoothed = rte
        
    if len(relative_orientation_error_deque)>0 and roe > 10*np.mean(relative_orientation_error_deque):
        roe_smoothed = 0.001 * roe + 0.999 * np.mean(relative_orientation_error_deque)
    else: roe_smoothed = roe
    
    relative_position_error_deque.append(rte_smoothed)
    relative_orientation_error_deque.append(roe_smoothed)
    
    #* Relative RMSE
    relative_RMSE_position.append(rte**2)
    relative_RMSE_orientation.append(roe**2)
              
    #** Store data
    # .......................................................................................... #     
    gt_positions.append(T_W_I1_gt.t)
    gt_rotations.append(T_W_I1_gt.R) 
    estimated_positions.append(T_W_I1_est.t)
    estimated_rotations.append(T_W_I1_est.R)
    
    rte_array.append(rte_smoothed)
    roe_array.append(roe_smoothed)
    
    ate_x.append(abs_T_error.t[0])
    ate_x_bounds.append(x_bound)
    ate_y.append(abs_T_error.t[1])
    ate_y_bounds.append(y_bound)
    ate_z.append(abs_T_error.t[2])
    ate_z_bounds.append(z_bound)
    aoe_roll.append(R2euler(abs_T_error.R)[0])
    aoe_roll_bounds.append(roll_bound)
    aoe_pitch.append(R2euler(abs_T_error.R)[1])
    aoe_pitch_bounds.append(pitch_bound)
    aoe_yaw.append(R2euler(abs_T_error.R)[2])
    aoe_yaw_bounds.append(yaw_bound)
    
    
    #** Logging
    # .......................................................................................... #    
    # rr.set_time_sequence('frame', i)

    # rr.log('world/gt_trajectory_point', rr.Points3D(gt_positions, colors=[[0, 200, 0]], radii=rr_trajectory_radii))
    # rr.log('world/estimated_trajectory', rr.Points3D(estimated_positions, colors=[[0, 0, 255]], radii=rr_trajectory_radii))
    # rr.log('/world/camera_gt', rr.Transform3D(translation=T_W_I1_gt.t, mat3x3=T_W_I1_gt.R, axis_length=rr_axis_length))
    # rr.log('/world/camera_vio', rr.Transform3D(translation=T_W_I1_est.t, mat3x3=T_W_I1_est.R, axis_length=rr_axis_length)) 
    # rr.log('world/estimated_world_points', rr.Points3D(msckf.estimated_world_points, colors=[200, 200, 0], radii=rr_point_radii))
    # rr.log('world/current_processed_world_points', rr.Points3D(msckf.currently_processed_world_points, colors=[[200, 0, 150]], radii=rr_point_radii + 0.01))
    # rr.log('world/imu_position_covariance', rr.Ellipsoids3D(centers=[T_W_I1_gt.t], half_sizes=[3*sigmas_position], colors=[[255, 0, 0]], fill_mode=3))
        
    # rr.log('relative_translation_error/t', rr.Scalar(rte_smoothed))
    # rr.log('relative_orientation_error/r', rr.Scalar(roe_smoothed))
    
    # rr.log('absolute_translation_error/x', rr.Scalar(abs_T_error.t[0]))
    # rr.log('absolute_translation_error/xlb', rr.Scalar(-x_bound))
    # rr.log('absolute_translation_error/xub', rr.Scalar(x_bound))
    
    # rr.log('absolute_translation_error/y', rr.Scalar(abs_T_error.t[1]))
    # rr.log('absolute_translation_error/ylb', rr.Scalar(-y_bound))
    # rr.log('absolute_translation_error/yub', rr.Scalar(y_bound))
    
    # rr.log('absolute_translation_error/z', rr.Scalar(abs_T_error.t[2]))
    # rr.log('absolute_translation_error/zlb', rr.Scalar(-z_bound))
    # rr.log('absolute_translation_error/zub', rr.Scalar(+z_bound))
    
    # rr.log('absolute_orientation_error/roll', rr.Scalar(R2euler(abs_T_error.R)[0]))
    # rr.log('absolute_orientation_error/rolllb', rr.Scalar(-roll_bound))
    # rr.log('absolute_orientation_error/rollub', rr.Scalar(roll_bound))
    
    # rr.log('absolute_orientation_error/pitch', rr.Scalar(R2euler(abs_T_error.R)[1]))
    # rr.log('absolute_orientation_error/pitchlb', rr.Scalar(-pitch_bound))
    # rr.log('absolute_orientation_error/pitchub', rr.Scalar(pitch_bound))
    
    # rr.log('absolute_orientation_error/yaw', rr.Scalar(R2euler(abs_T_error.R)[2]))
    # rr.log('absolute_orientation_error/yawlb', rr.Scalar(-yaw_bound))
    # rr.log('absolute_orientation_error/yawub', rr.Scalar(yaw_bound))

    # rr.log('msckf/features', rr.Scalar(len(msckf.features)))
    # rr.log('msckf/camera_states', rr.Scalar(len(msckf.state.cameras)))

    T_W_I0_gt = T_W_I1_gt
    T_W_I0_est = T_W_I1_est
    

rmse_position    = np.round(np.sqrt(np.mean(relative_RMSE_position)), 8)
rmse_orientation = np.round(np.sqrt(np.mean(relative_RMSE_orientation)), 8)
mean_rte         = np.round(np.mean(rte_array),8)
std_rte          = np.round(np.std(rte_array),8)
mean_roe         = np.round(np.mean(roe_array),8)
std_roe          = np.round(np.std(roe_array),8)

mean_loop_time                     = np.round(np.mean(loop_time), 8)
mean_loop_time_with_camera         = np.round(np.mean(loop_time_with_camera), 8)
mean_loop_time_without_camera      = np.round(np.mean(loop_time_without_camera), 8)
mean_loop_frequency                = np.round(1/mean_loop_time, 8)
mean_loop_frequency_with_camera    = np.round(1/mean_loop_time_with_camera, 8)
mean_loop_frequency_without_camera = np.round(1/mean_loop_time_without_camera, 8)

str_results = f'VIO:\n'
str_results += f' - RTE %:\n'
str_results += f'   - Mean:           {mean_rte} - {mean_rte*100} %\n'
str_results += f'   - Std:            {std_rte} - {std_rte*100} %\n'
str_results += f' - ROE %:\n'   
str_results += f'   - Mean:           {mean_roe} - {mean_roe*100} %\n'
str_results += f'   - Std:            {std_roe} - {std_roe*100} %\n'
str_results += f' - RMSE %:\n'   
str_results += f'   - Position:       {rmse_position}\n'
str_results += f'   - Orientation:    {rmse_orientation}\n'
str_results += f' - Profiling:\n'
str_results += f'   - Mean:           {mean_loop_time} s - {mean_loop_frequency} Hz\n'
str_results += f'   - With Camera:    {mean_loop_time_with_camera} s - {mean_loop_frequency_with_camera} Hz\n'
str_results += f'   - Without Camera: {mean_loop_time_without_camera} s - {mean_loop_frequency_without_camera} Hz\n'

print(str_results)
with open(f'{experiment_folder}/results.txt', 'w') as f: f.write(str_results)

#** Plot
fix, ax = plt.subplots(2, 1, figsize=(6.6, 6), dpi=500, sharex='col')
ax[0].plot(rte_array, label='RTE', color='tab:blue', linewidth=1.5)
ax[0].set_title('Relative Translation Error %', fontsize=16)
ax[0].grid(True, color='gray', linestyle='-', linewidth=0.2)

ax[1].plot(roe_array, label='ROE', color='tab:blue', linewidth=1.5)
ax[1].set_title('Relative Orientation Error %', fontsize=16)
ax[1].set_xlabel('Frame', fontsize=12)
ax[1].grid(True, color='gray', linestyle='-', linewidth=0.2)

plt.tight_layout()
if save_results: plt.savefig(f'{experiment_folder}/RTE_ROE.png')

fig, ax = plt.subplots(2, 3, figsize=(18, 6), dpi=500, sharey='row', sharex='col')

ax[0, 0].plot(ate_x, label='ATE x', color='tab:red', linewidth=1.5)
ax[0, 0].plot(np.array(ate_x_bounds), label='3σ Bounds', linestyle='--', color='tab:red', linewidth=0.9)
ax[0, 0].plot(-np.array(ate_x_bounds), linestyle='--', color='tab:red', linewidth=0.9)
ax[0, 0].set_title('Absolute Translation Error x [m]', fontsize=16)
ax[0, 0].grid(True, color='gray', linestyle='-', linewidth=0.2)
ax[0, 0].legend(loc='upper left', fontsize=10)

ax[0, 1].plot(ate_y, label='ATE y', color='tab:green', linewidth=1.5)
ax[0, 1].plot(np.array(ate_y_bounds), label='3σ Bounds', linestyle='--', color='tab:green', linewidth=0.9)
ax[0, 1].plot(-np.array(ate_y_bounds), linestyle='--', color='tab:green', linewidth=0.9)
ax[0, 1].set_title('Absolute Translation Error y [m]', fontsize=16)
ax[0, 1].grid(True, color='gray', linestyle='-', linewidth=0.2)
ax[0, 1].legend(loc='upper left', fontsize=10)

ax[0, 2].plot(ate_z, label='ATE z', color='tab:blue', linewidth=1.5)
ax[0, 2].plot(np.array(ate_z_bounds), label='3σ Bounds', linestyle='--', color='tab:blue', linewidth=0.9)
ax[0, 2].plot(-np.array(ate_z_bounds), linestyle='--', color='tab:blue', linewidth=0.9)
ax[0, 2].set_title('Absolute Translation Error z [m]', fontsize=16)
ax[0, 2].grid(True, color='gray', linestyle='-', linewidth=0.2)
ax[0, 2].legend(loc='upper left', fontsize=10)

ax[1, 0].plot(aoe_roll, label='AOE roll', color='tab:red', linewidth=1.5)
ax[1, 0].plot(np.array(aoe_roll_bounds), label='3σ Bounds', linestyle='--', color='tab:red', linewidth=0.9)
ax[1, 0].plot(-np.array(aoe_roll_bounds), linestyle='--', color='tab:red', linewidth=0.9)
ax[1, 0].set_title('Absolute Orientation Error roll [rad]', fontsize=16)
ax[1, 0].set_xlabel('Frame', fontsize=12)
ax[1, 0].grid(True, color='gray', linestyle='-', linewidth=0.2)
ax[1, 0].legend(loc='upper left', fontsize=10)

ax[1, 1].plot(aoe_pitch, label='AOE pitch', color='tab:green', linewidth=1.5)
ax[1, 1].plot(np.array(aoe_pitch_bounds), label='3σ Bounds', linestyle='--', color='tab:green', linewidth=0.9)
ax[1, 1].plot(-np.array(aoe_pitch_bounds), linestyle='--', color='tab:green', linewidth=0.9)
ax[1, 1].set_title('Absolute Orientation Error pitch [rad]', fontsize=16)
ax[1, 1].set_xlabel('Frame', fontsize=12)
ax[1, 1].grid(True, color='gray', linestyle='-', linewidth=0.2)
ax[1, 1].legend(loc='upper left', fontsize=10)

ax[1, 2].plot(aoe_yaw, label='AOE yaw', color='tab:blue', linewidth=1.5)
ax[1, 2].plot(np.array(aoe_yaw_bounds), label='3σ Bounds', linestyle='--', color='tab:blue', linewidth=0.9)
ax[1, 2].plot(-np.array(aoe_yaw_bounds), linestyle='--', color='tab:blue', linewidth=0.9)
ax[1, 2].set_title('Absolute Orientation Error yaw [rad]', fontsize=16)
ax[1, 2].set_xlabel('Frame', fontsize=12)
ax[1, 2].grid(True, color='gray', linestyle='-', linewidth=0.2)
ax[1, 2].legend(loc='upper left', fontsize=10)

plt.tight_layout()
if save_results: plt.savefig(f'{experiment_folder}/ATE_AOE.png')
# plt.show()