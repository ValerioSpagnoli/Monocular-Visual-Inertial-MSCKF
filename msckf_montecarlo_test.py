import sys
sys.path.append('../../')

import numpy as np
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import cv2
from collections import deque
import time

from src.utils.geometry import *
from src.utils.visualization_utils import *
from src.msckf.IMU import *
from src.msckf.MSCKF import *
from src.msckf.Camera import *
from dataset.tools import parser
from dataset.tools.dataset_generators.photorealistic_generator import PhotorealisticGenerator

max_frames = 3000
montecarlo_runs = 50

BASE_DATASET_PATH = '/home/valeriospagnoli/Thesis/vio/dataset'
source = 'peringlab' # ['synthetic', 'peringlab', 'tartanair']
sequence = 'deer_running'  

camera_info = pd.read_csv(f'{BASE_DATASET_PATH}/{source}/camera_info.csv')
T_W_C = Isometry3D(np.array([[0,0,1], [-1,0,0], [0,-1,0]]),  np.zeros(3))
K = np.array([[camera_info.iloc[0]['fx'], 0, camera_info.iloc[0]['px']],
            [0, camera_info.iloc[0]['fy'], camera_info.iloc[0]['py']],
            [0, 0, 1]])

width = camera_info.iloc[0]['w']
height = camera_info.iloc[0]['h']

experiment_folder = f'/home/valeriospagnoli/Thesis/vio/experiments/{source}/{sequence}'

accelerometer_noise_density = 0.01
gyroscope_noise_density = 0.001
accelerometer_random_walk = 0.001
gyroscope_random_walk = 0.0001

def run_simulation(run=0):
    
    #** Generation
    # ========================================================================================== #
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

    #** Initialize the test
    # ========================================================================================== #
    last_camera_time_index = 1

    T_W_I0_gt = Isometry3D(np.eye(3), np.zeros(3))
    T_W_I0_est_vio = Isometry3D(np.eye(3), np.zeros(3))

    relative_RMSE_translation_array = np.zeros(max_frames)
    relative_RMSE_orientation_array = np.zeros(max_frames)
    NEES_array = np.zeros(max_frames)
    NEES_position_array = np.zeros(max_frames)
    NEES_orientation_array = np.zeros(max_frames)
    NEES_array_x = np.zeros(max_frames)
    NEES_array_y = np.zeros(max_frames)
    NEES_array_z = np.zeros(max_frames)
    NEES_array_roll = np.zeros(max_frames)
    NEES_array_pitch = np.zeros(max_frames)
    NEES_array_yaw = np.zeros(max_frames)
    
    covariance_pose = np.zeros((6, 6))
    error_pose = np.zeros((1, 6))
    
    RPE_position_deque = deque(maxlen=5)
    RPE_orientation_deque = deque(maxlen=5)
    
    loop_time_sum = 0
    loop_time_with_camera = 0
    counter_camera_measurements = 0
    start_simulation_time = time.time()
    
    #** Loop
    # ========================================================================================== #
    for i in tqdm(range(0, max_frames)):             
        if i > len(imu_data)-1: break
        if last_camera_time_index+1 > len(camera_images)-1: break
               
        start_loop_time = time.time()
        start_loop_time_with_camera = time.time()
        is_there_camera_measurement = False
        
        #** Get measurements
        # .......................................................................................... #
        imu_timestamp = imu_data.iloc[i]['timestamp']
        camera_timestamp = camera_images.iloc[last_camera_time_index]['timestamp']
        angular_velocity = np.array([imu_data.iloc[i]['wx'], imu_data.iloc[i]['wy'], imu_data.iloc[i]['wz']])
        linear_acceleration = np.array([imu_data.iloc[i]['ax'], imu_data.iloc[i]['ay'], imu_data.iloc[i]['az']])
        imu_measurement = IMUMeasurement(imu_timestamp, angular_velocity, linear_acceleration)
                
        #** Test VIO with MSCKF
        # .......................................................................................... #
        msckf.imu_callback(imu_measurement)
        if np.abs(np.round(imu_timestamp - camera_timestamp, 3)) < 0.00001:
            is_there_camera_measurement = True
            counter_camera_measurements += 1
            image_path = camera_images.iloc[last_camera_time_index]['image_path']
            last_camera_time_index += 1
            
            if source == 'synthetic':
                keypoints, descriptors, scores = parser.extract_synthetic_camera_measurements(image_path)
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

        
        #* Estimate VIO
        T_W_I1_est_vio = msckf.state.imu.T_W_Ii
        T_I0_I1_est_vio = T_W_I0_est_vio.inv() * T_W_I1_est_vio
        
        # Absolute Pose Error
        abs_T_error_vio = T_W_I1_gt.inv() * T_W_I1_est_vio
        
        # Relative Pose Error
        rel_T_error_vio = T_I0_I1_gt.inv() * T_I0_I1_est_vio
        relative_translation_error_vio = np.linalg.norm(rel_T_error_vio.t)
        trace_R = np.trace(rel_T_error_vio.R)
        relative_orientation_error_vio = np.arccos(np.clip((trace_R - 1) / 2.0, -1.0, 1.0))
        
        if len(RPE_position_deque)>0 and relative_translation_error_vio > 2*np.mean(RPE_position_deque):
            relative_translation_error_vio = 0.1 * relative_translation_error_vio + 0.9 * np.mean(RPE_position_deque)
            
        if len(RPE_orientation_deque)>0 and relative_orientation_error_vio > 5*np.mean(RPE_orientation_deque):
            relative_orientation_error_vio = 0.1 * relative_orientation_error_vio + 0.9 * np.mean(RPE_orientation_deque)
        
        RPE_position_deque.append(relative_translation_error_vio)
        RPE_orientation_deque.append(relative_orientation_error_vio)
        
        relative_RMSE_translation_array[i] = relative_translation_error_vio**2
        relative_RMSE_orientation_array[i] = relative_orientation_error_vio**2

        # NEES: Normalized Estimation Error Squared
        # Measures the discrepancy between the true state (or ground truth) and the estimated state, weighted by the filter’s estimated covariance.
        # A NEES value that matches the expected chi-square distribution (with degrees of freedom equal to the state dimension under evaluation) indicates that the filter’s uncertainty is realistic. 
        # - If the NEES value is too low:  the estimated uncertainty (P) is overestimated  (the filter is underconfident about the state). -> IMU sigma values are too high.
        # - If the NEES value is too high: the estimated uncertainty (P) is underestimated (the filter is overconfident about the state).  -> IMU sigma values are too low.
        
        absolute_pose_error = np.concatenate([R2axisAngle(abs_T_error_vio.R), abs_T_error_vio.t])
        P_orientation = msckf.state.covariance[:3, :3] + 1e-6 * np.eye(3)
        P_position = msckf.state.covariance[12:15, 12:15] + 1e-6 * np.eye(3)
        P_orientation_position = msckf.state.covariance[:3, 12:15] + 1e-6 * np.eye(3)
        P_pose = np.block([[P_orientation, P_orientation_position], [P_orientation_position.T, P_position]]) + 1e-6 * np.eye(6)
        NEES_metric = absolute_pose_error.T @ np.linalg.inv(P_pose) @ absolute_pose_error
        NEES_array[i] += NEES_metric
        
        NEES_position = abs_T_error_vio.t.T @ np.linalg.inv(P_position) @ abs_T_error_vio.t
        NEES_orientation = R2axisAngle(abs_T_error_vio.R).T @ np.linalg.inv(P_orientation) @ R2axisAngle(abs_T_error_vio.R)
        NEES_position_array[i] = NEES_position
        NEES_orientation_array[i] = NEES_orientation
        
        NEES_x = (abs_T_error_vio.t[0] ** 2) / P_position[0, 0]
        NEES_y = (abs_T_error_vio.t[1] ** 2) / P_position[1, 1]
        NEES_z = (abs_T_error_vio.t[2] ** 2) / P_position[2, 2]
        NEES_roll  = (R2axisAngle(abs_T_error_vio.R)[0] ** 2) / P_orientation[0, 0]
        NEES_pitch = (R2axisAngle(abs_T_error_vio.R)[1] ** 2) / P_orientation[1, 1]
        NEES_yaw   = (R2axisAngle(abs_T_error_vio.R)[2] ** 2) / P_orientation[2, 2]
        
        NEES_array_x[i] = NEES_x
        NEES_array_y[i] = NEES_y
        NEES_array_z[i] = NEES_z
        NEES_array_roll[i] = NEES_roll
        NEES_array_pitch[i] = NEES_pitch
        NEES_array_yaw[i] = NEES_yaw
        
        covariance_pose += P_pose
        pose_error_squared = np.concatenate([R2axisAngle(abs_T_error_vio.R), abs_T_error_vio.t]) ** 2
        error_pose += pose_error_squared
                                                    
        T_W_I0_gt = T_W_I1_gt
        T_W_I0_est_vio = T_W_I1_est_vio  
        
        end_loop_time = time.time() 
        end_loop_time_with_camera = time.time()
        
        loop_time_sum += end_loop_time - start_loop_time
        if is_there_camera_measurement:
            loop_time_with_camera += end_loop_time_with_camera - start_loop_time_with_camera        
        
    end_simulation_time = time.time()  
    
    mean_covariance_pose = covariance_pose / max_frames
    mean_error_pose = error_pose / max_frames
    
    simulation_time = end_simulation_time - start_simulation_time
    mean_loop_time = loop_time_sum / max_frames
    mean_loop_time_with_camera = loop_time_with_camera / counter_camera_measurements
        
    return  relative_RMSE_translation_array, relative_RMSE_orientation_array, \
            NEES_array, NEES_position_array, NEES_orientation_array, \
            NEES_array_x, NEES_array_y, NEES_array_z, NEES_array_roll, NEES_array_pitch, NEES_array_yaw, \
            mean_covariance_pose, mean_error_pose, \
            simulation_time, mean_loop_time, mean_loop_time_with_camera


#** Monte Carlo
# ========================================================================================== #

w = 12
while True:        
        
    relative_RMSE_translation_array = np.zeros(max_frames)
    relative_RMSE_orientation_array = np.zeros(max_frames)
    NEES_array = np.zeros(max_frames)
    NEES_array_position = np.zeros(max_frames)
    NEES_array_orientation = np.zeros(max_frames)
    NEES_array_x = np.zeros(max_frames)
    NEES_array_y = np.zeros(max_frames)
    NEES_array_z = np.zeros(max_frames)
    NEES_array_roll = np.zeros(max_frames)
    NEES_array_pitch = np.zeros(max_frames)
    NEES_array_yaw = np.zeros(max_frames)

    covariance_pose_runs = np.zeros((6, 6))
    error_pose_runs = np.zeros((1, 6))

    simulation_time_runs = []
    mean_loop_frequency_runs = []
    mean_loop_frequency_with_camera_runs = []


    for k in range(montecarlo_runs):
        print(f'\nRun {k+1}/{montecarlo_runs}')   
        relative_RMSE_translation_array_run, relative_RMSE_orientation_array_run, \
        NEES_array_run, NEES_array_position_run, NEES_array_orientation_run, \
        NEES_array_x_run, NEES_array_y_run, NEES_array_z_run, NEES_array_roll_run, NEES_array_pitch_run, NEES_array_yaw_run, \
        covariance_pose, error_pose, \
        simulation_time, mean_loop_time, mean_loop_time_with_camera = run_simulation(k)
    
        relative_RMSE_translation_array += relative_RMSE_translation_array_run
        relative_RMSE_orientation_array += relative_RMSE_orientation_array_run
        NEES_array += NEES_array_run
        NEES_array_position += NEES_array_position_run
        NEES_array_orientation += NEES_array_orientation_run
        NEES_array_x += NEES_array_x_run
        NEES_array_y += NEES_array_y_run
        NEES_array_z += NEES_array_z_run
        NEES_array_roll += NEES_array_roll_run
        NEES_array_pitch += NEES_array_pitch_run
        NEES_array_yaw += NEES_array_yaw_run
        
        covariance_pose_runs += covariance_pose
        error_pose_runs += error_pose
        
        simulation_time_runs.append(simulation_time)
        mean_loop_frequency_runs.append(1/mean_loop_time)
        mean_loop_frequency_with_camera_runs.append(1/mean_loop_time_with_camera)

    relative_RMSE_translation_array /= montecarlo_runs
    relative_RMSE_orientation_array /= montecarlo_runs
    relative_RMSE_translation_array = np.sqrt(relative_RMSE_translation_array)
    relative_RMSE_orientation_array = np.sqrt(relative_RMSE_orientation_array)

    NEES_array /= montecarlo_runs
    NEES_6dof_lb = (1/montecarlo_runs) * chi2.ppf(0.05/2, 6*montecarlo_runs)
    NEES_6dof_ub = (1/montecarlo_runs) * chi2.ppf(1-(0.05/2), 6*montecarlo_runs)

    NEES_array_position = NEES_array_position / montecarlo_runs
    NEES_array_orientation = NEES_array_orientation / montecarlo_runs
    NEES_3dof_lb = (1/montecarlo_runs) * chi2.ppf(0.05/2, 3*montecarlo_runs)
    NEES_3dof_ub = (1/montecarlo_runs) * chi2.ppf(1-(0.05/2), 3*montecarlo_runs)

    NEES_array_x = NEES_array_x / montecarlo_runs
    NEES_array_y = NEES_array_y / montecarlo_runs
    NEES_array_z = NEES_array_z / montecarlo_runs
    NEES_array_roll = NEES_array_roll / montecarlo_runs
    NEES_array_pitch = NEES_array_pitch / montecarlo_runs
    NEES_array_yaw = NEES_array_yaw / montecarlo_runs
    NEES_1dof_lb = (1/montecarlo_runs) * chi2.ppf(0.05/2, montecarlo_runs)
    NEES_1dof_ub = (1/montecarlo_runs) * chi2.ppf(1-(0.05/2), montecarlo_runs)
    
    covariance_pose_runs /= montecarlo_runs
    error_pose_runs /= montecarlo_runs

    simulation_time_avg = np.mean(simulation_time_runs)
    mean_loop_frequency_avg = np.mean(mean_loop_frequency_runs)
    mean_loop_frequency_with_camera_avg = np.mean(mean_loop_frequency_with_camera_runs)

    print(f'\nMean Simulation Time:            {simulation_time_avg} s')
    print(f'Mean Loop Frequency:               {mean_loop_frequency_avg} Hz')
    print(f'Mean Loop Frequency (With Camera): {mean_loop_frequency_with_camera_avg} Hz')

    #** Plot
    # ========================================================================================== #
    fig, axs = plt.subplots(2, 1, figsize=(6.6, 9), dpi=500, sharex='col')

    axs[0].plot(relative_RMSE_translation_array, label='Translation RMSE', color='tab:blue', linewidth=1.5)
    axs[0].set_title('Average Relative Translation RMSE [m]', fontsize=16)
    axs[0].grid(True, color='gray', linestyle='-', linewidth=0.2)

    axs[1].plot(relative_RMSE_orientation_array, label='Orientation RMSE', color='tab:blue', linewidth=1.5)
    axs[1].set_title('Average Relative Orientation RMSE [rad]', fontsize=16)
    axs[1].grid(True, color='gray', linestyle='-', linewidth=0.2)

    plt.tight_layout()
    plt.savefig(f'{experiment_folder}/montecarlo_test_rmse_{w}.png')

    fig, axs = plt.subplots(3, 3, figsize=(18, 10), dpi=500, sharex='col')

    axs[0][0].plot(NEES_array, label='NEES', color='tab:blue', linewidth=1.5)
    axs[0][0].axhline(y=NEES_6dof_lb, color='tab:red', linestyle='--', label='Lower Bound', linewidth=0.9)
    axs[0][0].axhline(y=NEES_6dof_ub, color='tab:red', linestyle='--', label='Upper Bound', linewidth=0.9)
    axs[0][0].set_title('Average Pose NEES', fontsize=16)
    axs[0][0].set_xlabel('Frame', fontsize=12)
    axs[0][0].legend(loc='upper left', fontsize=10)
    axs[0][0].grid(True, color='gray', linestyle='-', linewidth=0.2)

    axs[1][0].plot(NEES_array_position, label='NEES Position', color='tab:blue', linewidth=1.5)
    axs[1][0].axhline(y=NEES_3dof_lb, color='tab:red', linestyle='--', label='Lower Bound', linewidth=0.9)
    axs[1][0].axhline(y=NEES_3dof_ub, color='tab:red', linestyle='--', label='Upper Bound', linewidth=0.9)
    axs[1][0].set_title('Average Position NEES', fontsize=16)
    axs[1][0].set_xlabel('Frame', fontsize=12)
    axs[1][0].legend(loc='upper left', fontsize=10)
    axs[1][0].grid(True, color='gray', linestyle='-', linewidth=0.2)

    axs[2][0].plot(NEES_array_orientation, label='NEES Orientation', color='tab:blue', linewidth=1.5)
    axs[2][0].axhline(y=NEES_3dof_lb, color='tab:red', linestyle='--', label='Lower Bound', linewidth=0.9)  
    axs[2][0].axhline(y=NEES_3dof_ub, color='tab:red', linestyle='--', label='Upper Bound', linewidth=0.9)
    axs[2][0].set_title('Average Orientation NEES', fontsize=16)
    axs[2][0].set_xlabel('Frame', fontsize=12)
    axs[2][0].legend(loc='upper left', fontsize=10)
    axs[2][0].grid(True, color='gray', linestyle='-', linewidth=0.2)

    axs[0][1].plot(NEES_array_x, label='NEES X', color='tab:blue', linewidth=1.5)
    axs[0][1].axhline(y=NEES_1dof_lb, color='tab:red', linestyle='--', label='Lower Bound', linewidth=0.9)
    axs[0][1].axhline(y=NEES_1dof_ub, color='tab:red', linestyle='--', label='Upper Bound', linewidth=0.9)
    axs[0][1].set_title('Average X NEES', fontsize=16)
    axs[0][1].set_xlabel('Frame', fontsize=12)
    axs[0][1].legend(loc='upper left', fontsize=10)
    axs[0][1].grid(True, color='gray', linestyle='-', linewidth=0.2)

    axs[1][1].plot(NEES_array_y, label='NEES Y', color='tab:blue', linewidth=1.5)
    axs[1][1].axhline(y=NEES_1dof_lb, color='tab:red', linestyle='--', label='Lower Bound', linewidth=0.9)
    axs[1][1].axhline(y=NEES_1dof_ub, color='tab:red', linestyle='--', label='Upper Bound', linewidth=0.9)
    axs[1][1].set_title('Average Y NEES', fontsize=16)
    axs[1][1].set_xlabel('Frame', fontsize=12)
    axs[1][1].legend(loc='upper left', fontsize=10)
    axs[1][1].grid(True, color='gray', linestyle='-', linewidth=0.2)

    axs[2][1].plot(NEES_array_z, label='NEES Z', color='tab:blue', linewidth=1.5)
    axs[2][1].axhline(y=NEES_1dof_lb, color='tab:red', linestyle='--', label='Lower Bound', linewidth=0.9)
    axs[2][1].axhline(y=NEES_1dof_ub, color='tab:red', linestyle='--', label='Upper Bound', linewidth=0.9)
    axs[2][1].set_title('Average Z NEES', fontsize=16)
    axs[2][1].set_xlabel('Frame', fontsize=12)
    axs[2][1].legend(loc='upper left', fontsize=10)
    axs[2][1].grid(True, color='gray', linestyle='-', linewidth=0.2)

    axs[0][2].plot(NEES_array_roll, label='NEES Roll', color='tab:blue', linewidth=1.5)
    axs[0][2].axhline(y=NEES_1dof_lb, color='tab:red', linestyle='--', label='Lower Bound', linewidth=0.9)
    axs[0][2].axhline(y=NEES_1dof_ub, color='tab:red', linestyle='--', label='Upper Bound', linewidth=0.9)
    axs[0][2].set_title('Average Roll NEES', fontsize=16)
    axs[0][2].set_xlabel('Frame', fontsize=12)
    axs[0][2].legend(loc='upper left', fontsize=10)
    axs[0][2].grid(True, color='gray', linestyle='-', linewidth=0.2)

    axs[1][2].plot(NEES_array_pitch, label='NEES Pitch', color='tab:blue', linewidth=1.5)
    axs[1][2].axhline(y=NEES_1dof_lb, color='tab:red', linestyle='--', label='Lower Bound', linewidth=0.9)
    axs[1][2].axhline(y=NEES_1dof_ub, color='tab:red', linestyle='--', label='Upper Bound', linewidth=0.9)
    axs[1][2].set_title('Average Pitch NEES', fontsize=16)
    axs[1][2].set_xlabel('Frame', fontsize=12)
    axs[1][2].legend(loc='upper left', fontsize=10)
    axs[1][2].grid(True, color='gray', linestyle='-', linewidth=0.2)

    axs[2][2].plot(NEES_array_yaw, label='NEES Yaw', color='tab:blue', linewidth=1.5)
    axs[2][2].axhline(y=NEES_1dof_lb, color='tab:red', linestyle='--', label='Lower Bound', linewidth=0.9)
    axs[2][2].axhline(y=NEES_1dof_ub, color='tab:red', linestyle='--', label='Upper Bound', linewidth=0.9)
    axs[2][2].set_title('Average Yaw NEES', fontsize=16)
    axs[2][2].set_xlabel('Frame', fontsize=12)
    axs[2][2].legend(loc='upper left', fontsize=10)
    axs[2][2].grid(True, color='gray', linestyle='-', linewidth=0.2)

    plt.tight_layout()
    plt.savefig(f'{experiment_folder}/montecarlo_test_nees_{w}.png')

    fig, axs = plt.subplots(2, 1, figsize=(6.6, 9), dpi=500, sharex='col')

    axs[0].plot(mean_loop_frequency_runs, label='Mean Loop Time', color='tab:blue', linewidth=1.5)
    axs[0].set_title('Average Loop Frequency [Hz]', fontsize=16)
    axs[0].set_xlabel('Run', fontsize=12)
    axs[0].grid(True, color='gray', linestyle='-', linewidth=0.2)

    axs[1].plot(mean_loop_frequency_with_camera_runs, label='Mean Loop Time (With Camera)', color='tab:blue', linewidth=1.5)
    axs[1].set_title('Average Loop Frequency if Camera Measurement [Hz]', fontsize=16)
    axs[1].set_xlabel('Run', fontsize=12)
    axs[1].grid(True, color='gray', linestyle='-', linewidth=0.2)

    min_x = 0
    max_x = montecarlo_runs - 1
    axs[1].set_xticks([i if i%2==0 else 0 for i in range(min_x, max_x + 1)])

    plt.tight_layout()
    plt.savefig(f'{experiment_folder}/profiling_{w}.png')


    error_norm = error_pose_runs / np.max(error_pose_runs)
    error_norm = error_norm.reshape(1, 6)

    cov_diag = np.diag(covariance_pose_runs)
    cov_diag_norm = cov_diag / np.max(cov_diag)
    cov_diag_norm = cov_diag_norm.reshape(1, 6)

    fig = plt.figure(figsize=(5, 3), dpi=500)

    gs = gridspec.GridSpec(
        2, 2, 
        width_ratios=[1, 0.05],
        height_ratios=[1, 1]
    )
    
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[:, 1])

    im_err = ax0.imshow(error_norm, cmap='Blues', vmin=0, vmax=1, aspect=1)
    ax0.set_title('Average Normalized Pose Error')
    ax0.set_xticks(np.arange(6))
    ax0.set_xticklabels(['roll', 'pitch', 'yaw', 'x', 'y', 'z'])
    ax0.set_yticks([0])
    ax0.set_yticklabels(['error'])

    im_cov = ax1.imshow(cov_diag_norm, cmap='Blues', vmin=0, vmax=1, aspect=1)
    ax1.set_title('Average Normalized Covariance Diagonal')
    ax1.set_xticks(np.arange(6))
    ax1.set_xticklabels(['roll', 'pitch', 'yaw', 'x', 'y', 'z'])
    ax1.set_yticks([0])
    ax1.set_yticklabels(['covariance'])

    cbar = fig.colorbar(im_cov, cax=cax, orientation='vertical')
    cbar.ax.set_ylabel('')

    # plt.tight_layout()
    plt.savefig(f'{experiment_folder}/error_covariance_pose_{w}.png', bbox_inches='tight')
        
    w += 1
    time.sleep(10)

    #plt.show()