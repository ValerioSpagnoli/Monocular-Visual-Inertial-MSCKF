import sys
sys.path.append('../../')

import signal
from urllib.parse import urlparse
import time
import os
import optuna
import gc

import numpy as np
import pandas as pd
import cv2

from src.utils.geometry import *
from src.msckf.IMU import *
from src.msckf.MSCKF import *
from src.msckf.Camera import *
from dataset.tools import parser

def input_with_timeout(prompt, timeout):
    print(prompt, end='', flush=True)
    def handler(signum, frame):
        raise TimeoutError
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        answer = input()
        signal.alarm(0)
        return answer
    except TimeoutError:
        signal.alarm(0)
        return None
    
    
BASE_DATASET_PATH = '/home/valeriospagnoli/Thesis/vio/dataset'

study_name = 'msckf_optimization_peringlab'
optuna_result_path = f'optimization_results/{study_name}.txt'
optuna_storage = f'sqlite:///optimization_results/{study_name}.db'

sequences = [
    # 'tartanair/ME000', 
    # 'tartanair/ME001', 
    # 'tartanair/ME003',
    # 'tartanair/ME005', 
    # 'tartanair/ME007'
    'peringlab/deer_mav_slow',
    'peringlab/deer_running'
    ]

num_trials = 50
max_frames_per_sequence = 3000


def objective(trial):
    # sigma_image = trial.suggest_float("sigma_image", 0.01, 0.2)
    # accelerometer_noise_density = trial.suggest_float("accelerometer_noise_density", 0.005, 0.1)
    # gyroscope_noise_density = trial.suggest_float("gyroscope_noise_density", 0.0005, 0.01)
    # accelerometer_random_walk = trial.suggest_float("accelerometer_random_walk", 0.0005, 0.01)
    # gyroscope_random_walk = trial.suggest_float("gyroscope_random_walk", 0.00005, 0.001)
    number_of_extracted_features = trial.suggest_int("number_of_extracted_features", 250, 350)
    min_cosine_similarity = trial.suggest_float("min_cosine_similarity", 0.87, 0.97)
    min_parallax = trial.suggest_float("min_parallax", 25, 50)
    epipolar_rejection_threshold = trial.suggest_float("epipolar_rejection_threshold", 0.001, 0.01)
    homography_rejection_threshold = trial.suggest_float("homography_rejection_threshold", 1, 10)
    min_number_of_frames_to_be_tracked = trial.suggest_int("min_number_of_frames_to_be_tracked", 2, 10)
    min_number_of_frames_to_be_lost = trial.suggest_int("min_number_of_frames_to_be_lost", 1, 10)
    max_number_of_camera_states = trial.suggest_int("max_number_of_camera_states", 30, 60)

    print(f'Trial {trial.number}')
    print(f' - Parameters:')
    # print(f' | * sigma_image:                        {trial.params["sigma_image"]}')
    # print(f' | * accelerometer_noise_density:        {trial.params["accelerometer_noise_density"]}')
    # print(f' | * gyroscope_noise_density:            {trial.params["gyroscope_noise_density"]}')
    # print(f' | * accelerometer_random_walk:          {trial.params["accelerometer_random_walk"]}')
    # print(f' | * gyroscope_random_walk:              {trial.params["gyroscope_random_walk"]}')
    print(f' | * number_of_extracted_features:       {trial.params["number_of_extracted_features"]}')
    print(f' | * min_cosine_similarity:              {trial.params["min_cosine_similarity"]}')
    print(f' | * min_parallax:                       {trial.params["min_parallax"]}')
    print(f' | * epipolar_rejection_threshold:       {trial.params["epipolar_rejection_threshold"]}')
    print(f' | * homography_rejection_threshold:     {trial.params["homography_rejection_threshold"]}')
    print(f' | * min_number_of_frames_to_be_tracked: {trial.params["min_number_of_frames_to_be_tracked"]}')
    print(f' | * min_number_of_frames_to_be_lost:    {trial.params["min_number_of_frames_to_be_lost"]}')
    print(f' | * max_number_of_camera_states:        {trial.params["max_number_of_camera_states"]}')
    print(f' |')
    
    start_trial = time.time()
    
    total_error_across_sequences = 0.0
    num_sequences = 0

    for seq in sequences:
        source, seq_name = seq.split('/')
        print(f' |-> Processing sequence {seq:<25} -> ', end='', flush=True)   
        start_sequence = time.time()
             
        #* Parse the camera info
        camera_info = pd.read_csv(f'{BASE_DATASET_PATH}/{source}/camera_info.csv')
        T_W_C = Isometry3D(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]), np.zeros(3))
        K = np.array([[camera_info.iloc[0]['fx'], 0, camera_info.iloc[0]['px']], [0, camera_info.iloc[0]['fy'], camera_info.iloc[0]['py']], [0, 0, 1]])
        width = camera_info.iloc[0]['w']
        height = camera_info.iloc[0]['h']

        #* Parse the sequence data
        current_parser = parser.Parser(source=source, sequence=seq_name, gt=False, initial_time_stamp=-1, final_time_stamp=-1)
        gt_trajectory = current_parser.extract_gt_trajectory()
        imu_data = current_parser.extract_imu()
        camera_images = current_parser.extract_images()

        #* Setup MSCKF with the current parameters
        msckf_parameters = MSCKFParameters(
            T_W_C = T_W_C,
            K = K,
            width = width,
            height = height,
            sigma_image = 0.1,
            
            only_imu = False,
            accelerometer_noise_density = 0.01,
            gyroscope_noise_density = 0.001,
            accelerometer_random_walk = 0.001,
            gyroscope_random_walk = 0.0001,
            W_gravity = np.array([0, 0, -9.81]),
            
            number_of_extracted_features = number_of_extracted_features,
            min_cosine_similarity = min_cosine_similarity,
            use_parallax = True,
            min_parallax = min_parallax,
            epipolar_rejection_threshold = epipolar_rejection_threshold,
            homography_rejection_threshold = homography_rejection_threshold,
            min_number_of_frames_to_be_tracked = min_number_of_frames_to_be_tracked,
            min_number_of_frames_to_be_lost = min_number_of_frames_to_be_lost,
            max_number_of_camera_states = max_number_of_camera_states
        )
        
        msckf = MSCKF(parameters=msckf_parameters)

        total_relative_pose_error = 0.0
        num_steps = 0
        
        last_camera_time_index = 1
        T_W_I0_gt = Isometry3D(np.eye(3), np.zeros(3))
        T_W_I0_est_vio = Isometry3D(np.eye(3), np.zeros(3))
        
        filtered_relative_translation_error = 0.0
        filtered_relative_orientation_error = 0.0
        low_pass_alpha = 0.1
        
        mean_time_per_step = 0.0
        for i in range(0, len(imu_data)):
            if max_frames_per_sequence > 0 and i > max_frames_per_sequence: break
            if last_camera_time_index+1 > len(camera_images)-1: break
            
            start_step = time.time()

            #* Test VIO with MSCKF
            imu_timestamp = imu_data.iloc[i]['timestamp']
            camera_timestamp = camera_images.iloc[last_camera_time_index]['timestamp']
            angular_velocity = np.array([imu_data.iloc[i]['wx'], imu_data.iloc[i]['wy'], imu_data.iloc[i]['wz']])
            linear_acceleration = np.array([imu_data.iloc[i]['ax'], imu_data.iloc[i]['ay'], imu_data.iloc[i]['az']])
            imu_measurement = IMUMeasurement(imu_timestamp, angular_velocity, linear_acceleration)
            
            msckf.imu_callback(imu_measurement)
            if np.abs(np.round(imu_timestamp - camera_timestamp, 3)) < 0.00001:
                image_path = camera_images.iloc[last_camera_time_index]['image_path']
                last_camera_time_index += 1
                image = cv2.imread(image_path)
                msckf.feature_callback(image)
                        
            #* Ground truth
            gt_transform = gt_trajectory.iloc[i]
            gt_position = np.array([gt_transform['T03'], gt_transform['T13'], gt_transform['T23']])
            gt_rotation = np.array([[gt_transform['T00'], gt_transform['T01'], gt_transform['T02']],
                                    [gt_transform['T10'], gt_transform['T11'], gt_transform['T12']],
                                    [gt_transform['T20'], gt_transform['T21'], gt_transform['T22']]])
            T_W_I1_gt = Isometry3D(gt_rotation, gt_position)
            T_I0_I1_gt = T_W_I0_gt.inv() * T_W_I1_gt

            #* VIO estimate
            T_W_I1_est_vio = msckf.state.imu.T_W_Ii
            T_I0_I1_est_vio = T_W_I0_est_vio.inv() * T_W_I1_est_vio

            #* Relative pose error
            rel_T_error = T_I0_I1_gt.inv() * T_I0_I1_est_vio
            relative_translation_error = np.linalg.norm(rel_T_error.t)
            relative_orientation_error = np.arccos(np.clip((np.trace(rel_T_error.R) - 1) / 2.0, -1.0, 1.0))
            
            filtered_relative_translation_error = low_pass_alpha * relative_translation_error + (1 - low_pass_alpha) * filtered_relative_translation_error
            filtered_relative_orientation_error = low_pass_alpha * relative_orientation_error + (1 - low_pass_alpha) * filtered_relative_orientation_error
            relative_pose_error = filtered_relative_translation_error + filtered_relative_orientation_error
                        
            #* NEES: Normalized Estimation Error Squared
            # Measures the discrepancy between the true state (or ground truth) and the estimated state, weighted by the filter’s estimated covariance.
            # A NEES value that matches the expected chi-square distribution (with degrees of freedom equal to the state dimension under evaluation) indicates that the filter’s uncertainty is realistic. 
            # - If the NEES value is too high, the filter is overconfident about the state (covariance underestimated -> IMU sigma values are too low), 
            # - If the NEES value is too low, the filter is underconfident about the state (covariance overestimated -> IMU sigma values are too high)., 
            abs_T_error_vio = T_W_I1_gt.inv() * T_W_I1_est_vio
            absolute_pose_error = np.concatenate([R2axisAngle(abs_T_error_vio.R), abs_T_error_vio.t])
            P_orientation = msckf.state.covariance[:3, :3]
            P_position = msckf.state.covariance[12:15, 12:15]
            P_orientation_position = msckf.state.covariance[:3, 12:15]
            P_pose = np.block([[P_orientation, P_orientation_position], [P_orientation_position.T, P_position]]) + 1e-6 * np.eye(6)
            NEES_metric = absolute_pose_error.T @ np.linalg.inv(P_pose) @ absolute_pose_error
            NEES_compare_lower = chi2.ppf(0.05/2, 6)
            NEES_compare_upper = chi2.ppf(1-(0.05/2), 6)
            NEES_compare_mean = (NEES_compare_lower + NEES_compare_upper) / 2
            NEES_error = np.abs(NEES_metric - NEES_compare_mean) ** 2
            
            #* Update statistics
            total_relative_pose_error += relative_pose_error
            num_steps += 1
            

            T_W_I0_gt = T_W_I1_gt
            T_W_I0_est_vio = T_W_I1_est_vio
            end_step = time.time()
            mean_time_per_step += end_step-start_step
            
        if num_steps > 0:
            sequence_error = total_relative_pose_error / num_steps
            total_error_across_sequences += sequence_error
            num_sequences += 1
            mean_time_per_step /= num_steps     
            
        del msckf    
        gc.collect()
        
        end_sequence = time.time()   
        print(f'error: {sequence_error:.6f} - total sequence time: {end_sequence-start_sequence:.2f} s - mean time per step: {mean_time_per_step:.4f} s [{1/mean_time_per_step:.2f} Hz]')
        print(f' |')
    
    end_trial = time.time()
    final_trial_error = total_error_across_sequences / num_sequences if num_sequences > 0 else float('inf')
    print(f' |-> Trial {trial.number} final objective value: {final_trial_error:.6f} - total trial time: {end_trial-start_trial:.2f} s\n')
    return final_trial_error

try:
    parsed_storage_url = urlparse(optuna_storage)
    db_path = parsed_storage_url.path[1:] if parsed_storage_url.path.startswith('/') else parsed_storage_url.path
    
    if os.path.exists(db_path):
        response = input(f'\nThe study {study_name} already exists. Do you want to to load it? (y/n): ')
        
        if response and response.lower() in ['y', 'yes']:
            print(f'Loading existing study {study_name} ...')
            study = optuna.load_study(study_name=study_name, storage=optuna_storage)
        else:
            response = input(f'\nDo you want to delete the existing study {study_name}? (y/n): ')
            
            if response and response.lower() in ['y', 'yes']:
                print(f'Deleting existing study {study_name} ...')
                study = optuna.delete_study(study_name=study_name, storage=optuna_storage)
                os.remove(db_path)
            else:
                print('Exiting ...')
                sys.exit(0)
            
            response = input(f'\nDo you want to create a new study {study_name}? (y/n): ')
            
            if response and response.lower() in ['y', 'yes']:
                print(f'Creating new study {study_name} ...\n')
                study = optuna.create_study(study_name=study_name, storage=optuna_storage, direction="minimize")
            else:
                print('Exiting...')
                sys.exit(0)
    
    else:
        study = optuna.create_study(study_name=study_name, storage=optuna_storage, direction="minimize")
    
    print(f'Starting optimization with {num_trials} trials ...')
    study.optimize(objective, n_trials=num_trials)
    
    with open(optuna_result_path, 'w') as f:
        f.write(f'Best objective value: {study.best_value}\n')
        f.write('Best parameters:\n')
        for key, value in study.best_params.items():
            f.write(f' - {key}: {value}\n')
        f.write('--------------------------------------------------\n')
        f.write('Trials:\n')
        for trial in study.trials:
            f.write(f' - {trial.params} - {trial.value}\n')
    sys.exit(0)

except KeyboardInterrupt:
    response = input_with_timeout("\nDo you want to close the session? (y/n): ", 5)
    
    if response and response.lower() in ['y', 'yes']:    
        print("\nExiting and saving study results ...")
        
        with open(optuna_result_path, 'w') as f:
            f.write(f'Best objective value: {study.best_value}\n')
            f.write('Best parameters:\n')
            for key, value in study.best_params.items():
                f.write(f' - {key}: {value}\n')
            f.write('--------------------------------------------------\n')
            f.write('Trials:\n')
            for trial in study.trials:
                f.write(f' - {trial.params} - {trial.value}\n')
        sys.exit(0)
    
    else:
        print("\nNo response or negative answer. Continuing computation ...")
        remaining_trials = num_trials - len(study.trials)
        print(f'Remaining trials: {remaining_trials}')
        print(f'number of trials: {num_trials} - len(study.trials): {len(study.trials)}')
        if remaining_trials > 0:
            print(f'Continuing optimization with {remaining_trials} trials ...')
            study = optuna.load_study(study_name=study_name, storage=optuna_storage)
            study.optimize(objective, n_trials=remaining_trials)
            
            with open(optuna_result_path, 'w') as f:
                f.write(f'Best objective value: {study.best_value}\n')
                f.write('Best parameters:\n')
                for key, value in study.best_params.items():
                    f.write(f' - {key}: {value}\n')
                f.write('--------------------------------------------------\n')
                f.write('Trials:\n')
                for trial in study.trials:
                    f.write(f' - {trial.params} - {trial.value}\n')
            sys.exit(0)