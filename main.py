import os
import argparse
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
from src.utils.rerun_visualizer import RerunVisualizer
from src.msckf.IMU import *
from src.msckf.MSCKF import *
from src.msckf.Camera import *
from dataset.tools import parser
from dataset.tools.dataset_generators.photorealistic_generator import PhotorealisticGenerator
from dataset.tools.dataset_generators.euroc_converter import EurocConverter

np.random.seed(42)

#** Parse command-line arguments
# ========================================================================================== #
# Arguments
# source: Dataset source (synthetic, peringlab, tartanair, euroc)
# sequence: Sequence name (P001, V1_01_easy, ...)
# max_frames: Maximum number of frames to process
# noise_level: Noise level for data generation (low, mid, high); ignored for euroc, which has real IMU data
# save_results: Save results to disk
# device: Torch device for XFeat (auto, cpu, mps, cuda)
# no_viewer: Do not stream to Rerun

parser_args = argparse.ArgumentParser(description='Run MSCKF VIO on dataset')
parser_args.add_argument('--source', type=str, default='tartanair', 
                        choices=['synthetic', 'peringlab', 'tartanair', 'euroc'],
                        help='Dataset source')
parser_args.add_argument('--sequence', type=str, default='P001',
                        help='Sequence name')
parser_args.add_argument('--max_frames', type=int, default=500,
                        help='Maximum number of frames to process')
parser_args.add_argument('--noise_level', type=str, default='mid',
                        choices=['low', 'mid', 'high'],
                        help='Noise level for data generation')
parser_args.add_argument('--save_results', action='store_true',
                        help='Save results to disk')
parser_args.add_argument('--epipolar_threshold', type=float, default=10.0,
                        help='Sampson distance [px] above which a match is rejected')
parser_args.add_argument('--no_viewer', action='store_true',
                        help='Do not stream to Rerun (faster, for evaluations)')
parser_args.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'mps', 'cuda'],
                        help='Torch device for XFeat feature extraction')
args = parser_args.parse_args()

BASE_DATASET_PATH = './data'
source = args.source
sequence = args.sequence
max_frames = args.max_frames
noise_level = args.noise_level if args.source != 'euroc' else 'real'
save_results = args.save_results
device = args.device
no_viewer = args.no_viewer

print(f'Running VIO on {source} dataset, sequence {sequence} with {noise_level} noise level')
print(f'Parameters:')
print(f'  - source: {source}')
print(f'  - sequence: {sequence}')
print(f'  - max_frames: {max_frames}')
print(f'  - noise_level: {noise_level}')
print(f'  - save_results: {save_results}')
print(f'  - device: {device}')
print(f'  - no_viewer: {no_viewer}')


if source == 'euroc':
    #* Real data: calibration from the dataset sensor.yaml files, images undistorted to the pinhole model K
    euroc = EurocConverter(sequence=sequence)
    T_W_C = euroc.T_I_C # camera pose in the IMU frame
    K = euroc.K
    width = euroc.width
    height = euroc.height
    
else:
    camera_info = pd.read_csv(f'{BASE_DATASET_PATH}/{source}/camera_info.csv')
    T_W_C = Isometry3D(np.array([[0,0,1], [-1,0,0], [0,-1,0]]),  np.zeros(3))
    K = np.array([[camera_info.iloc[0]['fx'], 0, camera_info.iloc[0]['px']],
                  [0, camera_info.iloc[0]['fy'], camera_info.iloc[0]['py']],
                  [0, 0, 1]])

    width = camera_info.iloc[0]['w']
    height = camera_info.iloc[0]['h']

experiment_folder = f'{BASE_DATASET_PATH}/{source}/{sequence}/{noise_level}_noise' if source != 'euroc' else f'{BASE_DATASET_PATH}/{source}/{sequence}/results'
if save_results:
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
        

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
    
elif noise_level == 'real':
    accelerometer_noise_density = euroc.accelerometer_noise_density
    gyroscope_noise_density = euroc.gyroscope_noise_density
    accelerometer_random_walk = euroc.accelerometer_random_walk
    gyroscope_random_walk = euroc.gyroscope_random_walk

#* Gravity convention: the generated datasets simulate an accelerometer measuring R^T (a + g),
#* while a real accelerometer measures R^T (a - g). Flipping W_gravity makes the filter consistent with real data.
W_gravity = np.array([0, 0, 9.81]) if source == 'euroc' else np.array([0, 0, -9.81])

#* Real IMU biases are not zero: estimate the gyroscope bias at the (static) start and give the filter
#* an initial uncertainty on the state, so the biases can be corrected by the visual updates
estimate_initial_gyroscope_bias = source == 'euroc'
if source == 'euroc':
    initial_covariance_diagonal = np.concatenate([np.full(3, 0.01**2),   # δθ [rad]
                                                  np.full(3, 0.005**2),  # δbg [rad/s]
                                                  np.full(3, 0.01**2),   # δv [m/s]
                                                  np.full(3, 0.2**2),    # δba [m/s^2]
                                                  np.zeros(3)])          # δp [m] (the start defines the origin)
else:
    initial_covariance_diagonal = np.zeros(15)

#** Photorealistic generator / EuRoC converter
# ========================================================================================== #
if source == 'euroc':
    euroc.process_data()
    
elif source != 'synthetic':
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
    W_gravity=W_gravity,
    estimate_initial_gyroscope_bias=estimate_initial_gyroscope_bias,
    initial_covariance_diagonal=initial_covariance_diagonal,
    
    #* Feature parameters
    number_of_extracted_features=300,
    min_cosine_similarity=0.95,
    use_parallax=True,
    min_parallax=45,
    epipolar_rejection_threshold=args.epipolar_threshold, # Sampson distance [px]
    homography_rejection_threshold=5,
    min_number_of_frames_to_be_tracked=4,
    min_number_of_frames_to_be_lost=2,
    max_number_of_camera_states=30,
    
    #* Feature extractor parameters
    device=device
)

msckf = MSCKF(parameters=msckf_parameters, rr=rr)


#** Rerun
# ========================================================================================== #
T_I_C = T_W_C if source == 'euroc' else msckf.state.imu.T_W_I.inv() * T_W_C # camera pose in the IMU frame
visualizer = RerunVisualizer(K=K, width=width, height=height, T_I_C=T_I_C,
                             save_path=f'{experiment_folder}/recording.rrd' if save_results else None, enabled=not no_viewer)


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

#* Running metrics shown in Rerun
ate_squared_sum = 0.0      # squared absolute position error [m^2]
aoe_squared_sum = 0.0      # squared absolute orientation error [rad^2]
inside_3sigma = 0          # steps with the position error inside the 3-sigma bounds on all axes
evaluated_steps = 0
traveled_distance = 0.0    # ground-truth path length [m]

#* Profiling
loop_time = []
loop_time_with_camera = []
loop_time_without_camera = []


def status_markdown(msckf: MSCKF, timestamp: float, loop_time: list) -> str:
    current_camera_id = msckf.state.imu.id
    visible = [f for f in msckf.features.values() if len(f.camera_indices) > 0 and f.camera_indices[-1] == current_camera_id]
    tracked = sum(1 for f in visible if len(f.keypoints) > 1)
    epipolar = sum(1 for _, reason in msckf.rejected_keypoints if reason == 'epipolar')
    homography = len(msckf.rejected_keypoints) - epipolar
    bg, ba = msckf.state.imu.gyroscope_bias, msckf.state.imu.accelerometer_bias
    return f'''
| Filter | Value |
|---|---|
| Time | {timestamp:.2f} s |
| Features in image (tracked / new) | {len(visible)} ({tracked} / {len(visible) - tracked}) |
| Features in the filter | {len(msckf.features)} |
| Rejected matches (epipolar / homography) | {epipolar} / {homography} |
| Features in last update (used / gated) | {len(msckf.updated_feature_indices)} / {len(msckf.gated_feature_indices)} |
| Camera states in window | {len(msckf.state.cameras)} |
| Gyroscope bias | [{bg[0]:+.4f}, {bg[1]:+.4f}, {bg[2]:+.4f}] rad/s |
| Accelerometer bias | [{ba[0]:+.3f}, {ba[1]:+.3f}, {ba[2]:+.3f}] m/s² |
| Mean loop rate | {1 / np.mean(loop_time):.1f} Hz |

**Image legend:** orange = used in update, green = tracked, light blue = new, magenta = gated, red = rejected match
'''

#** Initialize the test
# ========================================================================================== #
last_camera_time_index = 1
T_W_I0_gt = Isometry3D(np.eye(3), np.zeros(3))
T_W_I0_est = Isometry3D(np.eye(3), np.zeros(3))

#* Real datasets: ground truth lives in its own world frame (Vicon) and is not available for the first samples,
#* so it is aligned to the estimate at the first sample where it exists
align_gt = source == 'euroc'
T_est_gt = None # transform from the ground-truth world frame to the estimator world frame

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
    # Process the image once the IMU has been propagated up to its timestamp (real IMU and camera clocks never match exactly)
    if imu_timestamp >= camera_timestamp - 1e-6:
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
            if source == 'euroc': image = euroc.undistort(image)
            msckf.feature_callback(image)     
        
    
    end_loop = time.time()
    delta_time = end_loop - start_loop
    loop_time.append(delta_time)
    if is_there_camera_measurement: loop_time_with_camera.append(delta_time)
    else: loop_time_without_camera.append(delta_time)
    
    #** Errors
    # .......................................................................................... #
                
    #* Estimate VIO
    T_W_I1_est = msckf.state.imu.T_W_Ii
    
    #* Ground truth
    gt_transform = gt_trajectory.iloc[i]
    if gt_transform.isna().any(): # no ground truth for this sample: only log the estimate
        visualizer.set_time(i, imu_timestamp)
        visualizer.log_pose(T_W_I1_est, None)
        if is_there_camera_measurement: 
            visualizer.log_frame(msckf, image, T_W_I1_est, None)
            visualizer.log_text('*Waiting for ground truth...*', status_markdown(msckf, imu_timestamp, loop_time))
        T_W_I0_est = T_W_I1_est
        continue
    
    gt_position = np.array([gt_transform['T03'], gt_transform['T13'], gt_transform['T23']])
    gt_rotation = np.array([[gt_transform['T00'], gt_transform['T01'], gt_transform['T02']],
                            [gt_transform['T10'], gt_transform['T11'], gt_transform['T12']],
                            [gt_transform['T20'], gt_transform['T21'], gt_transform['T22']]])
    T_W_I1_gt = Isometry3D(gt_rotation, gt_position)
    
    if align_gt:
        if T_est_gt is None:
            T_est_gt = T_W_I1_est * T_W_I1_gt.inv()
            T_W_I0_gt, T_W_I0_est = T_est_gt * T_W_I1_gt, T_W_I1_est
        T_W_I1_gt = T_est_gt * T_W_I1_gt
    
    T_I0_I1_gt = T_W_I0_gt.inv() * T_W_I1_gt
    
    relative_translation_gt = np.linalg.norm(T_I0_I1_gt.t)
    relative_orientation_gt = np.arccos(np.clip((np.trace(T_I0_I1_gt.R) - 1) / 2.0, -1.0, 1.0))
    
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
    
    
    #* Running metrics
    evaluated_steps += 1
    ate_squared_sum += np.sum(abs_T_error.t**2)
    aoe_squared_sum += np.arccos(np.clip((np.trace(abs_T_error.R) - 1) / 2.0, -1.0, 1.0))**2
    inside_3sigma += int(np.all(np.abs(abs_T_error.t) <= 3*sigmas_position))
    traveled_distance += relative_translation_gt
    
    #** Logging
    # .......................................................................................... #    
    visualizer.set_time(i, imu_timestamp)
    visualizer.log_pose(T_W_I1_est, T_W_I1_gt)
    if is_there_camera_measurement:
        visualizer.log_frame(msckf, image, T_W_I1_est, T_W_I1_gt)
        
        ate_rmse = np.sqrt(ate_squared_sum / evaluated_steps)
        metrics = f'''
| Metric | Value |
|---|---|
| ATE position RMSE | {ate_rmse:.3f} m |
| ATE position (current) | {np.linalg.norm(abs_T_error.t):.3f} m |
| ATE orientation RMSE | {np.rad2deg(np.sqrt(aoe_squared_sum / evaluated_steps)):.2f} deg |
| Drift (ATE RMSE / distance) | {100 * ate_rmse / max(traveled_distance, 1e-9):.2f} % |
| Distance traveled (gt) | {traveled_distance:.2f} m |
| RTE mean | {100 * np.mean(rte_array):.2f} % |
| ROE mean | {100 * np.mean(roe_array):.3f} % |
| Position error inside 3σ | {100 * inside_3sigma / evaluated_steps:.1f} % of steps |
| Current error x / y / z | {abs_T_error.t[0]:+.3f} / {abs_T_error.t[1]:+.3f} / {abs_T_error.t[2]:+.3f} m |
| Current 3σ x / y / z | {x_bound:.3f} / {y_bound:.3f} / {z_bound:.3f} m |
'''
        visualizer.log_text(metrics, status_markdown(msckf, imu_timestamp, loop_time))
    
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
str_results += f' - ATE (evaluated over {evaluated_steps} steps, {traveled_distance:.2f} m):\n'
str_results += f'   - Position RMSE:  {np.sqrt(ate_squared_sum / max(evaluated_steps, 1)):.4f} m\n'
str_results += f'   - Orient. RMSE:   {np.rad2deg(np.sqrt(aoe_squared_sum / max(evaluated_steps, 1))):.3f} deg\n'
str_results += f'   - Inside 3σ:      {100 * inside_3sigma / max(evaluated_steps, 1):.1f} % of steps\n'
str_results += f' - Rejections:\n'
str_results += f'   - Epipolar:       {msckf.number_of_features_discarded_for_epipolar_test}\n'
str_results += f'   - Homography:     {msckf.number_of_features_discarder_for_homography_test}\n'
str_results += f'   - Gating (χ²):    {msckf.number_of_residuals_discarded_for_gasting_test}\n'
str_results += f' - Profiling:\n'
str_results += f'   - Mean:           {mean_loop_time} s - {mean_loop_frequency} Hz\n'
str_results += f'   - With Camera:    {mean_loop_time_with_camera} s - {mean_loop_frequency_with_camera} Hz\n'
str_results += f'   - Without Camera: {mean_loop_time_without_camera} s - {mean_loop_frequency_without_camera} Hz\n'

print(str_results)
if save_results:
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