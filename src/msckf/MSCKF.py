from src.msckf.Camera import *
from src.msckf.IMU import *
from src.msckf.FeatureExtractor import *
from src.utils.geometry import *
from src.utils.visualization_utils import *

from scipy.linalg import null_space
from scipy.stats import chi2

from dataclasses import dataclass, field
from typing import List, Dict
import rerun as rr

@dataclass
class MSCKFParameters:
    
    #* Camera parameters
    T_W_C: Isometry3D = field(default_factory=lambda: Isometry3D(np.array([[ 0,  0, 1],
                                                                            [-1,  0, 0],
                                                                            [ 0, -1, 0]]), np.array([0, 0, 0])))
    K: np.ndarray = field(default_factory=lambda: np.array([[180,   0, 320],
                                                            [  0, 180, 240],
                                                            [  0,   0,   1]]))
    width: int = 640
    height: int = 480
    sigma_image: float = 0.2
    
    #* IMU parameters
    only_imu: bool = False
    accelerometer_noise_density: float = 0.001
    accelerometer_random_walk: float = 0.00001
    gyroscope_noise_density: float = 0.0001
    gyroscope_random_walk: float = 0.000001
    W_gravity: np.ndarray = field(default_factory=lambda: np.array([0, 0, -9.81]))
    
    #* Feature parameters
    number_of_extracted_features: int = 256
    min_cosine_similarity: float = 0.82
    use_parallax: bool = True
    min_parallax: float = 20
    epipolar_rejection_threshold: float = 5
    homography_rejection_threshold: float = 5
    min_number_of_frames_to_be_lost: int = 1
    min_number_of_frames_to_be_tracked: int = 5
    max_number_of_camera_states: int = 30
    
    def to_str(self):
        return  f'\n\
T_W_C:\n{self.T_W_C.matrix()}\n\
K:\n{self.K}\n \
width: {self.width}\n \
height: {self.height}\n \
sigma_image: {self.sigma_image}\n \
only_imu: {self.only_imu}\n \
accelerometer_noise_density: {self.accelerometer_noise_density}\n \
accelerometer_random_walk: {self.accelerometer_random_walk}\n \
gyroscope_noise_density: {self.gyroscope_noise_density}\n \
gyroscope_random_walk: {self.gyroscope_random_walk}\n \
W_gravity: {self.W_gravity}\n \
number_of_extracted_features: {self.number_of_extracted_features}\n \
min_cosine_similarity: {self.min_cosine_similarity}\n \
use_parallax: {self.use_parallax}\n \
min_parallax: {self.min_parallax}\n \
epipolar_rejection_threshold: {self.epipolar_rejection_threshold}\n \
homography_rejection_threshold: {self.homography_rejection_threshold}\n \
min_number_of_frames_to_be_lost: {self.min_number_of_frames_to_be_lost}\n \
min_number_of_frames_to_be_tracked: {self.min_number_of_frames_to_be_tracked}\n \
max_number_of_camera_states: {self.max_number_of_camera_states}'
                            
    

@dataclass
class MSCKFState:
    imu: IMU = None
    cameras: Dict[int, Camera] = field(default_factory=dict)
    covariance: np.ndarray = field(default_factory=lambda: np.zeros((15, 15)))
    continuous_noise_covariance: np.ndarray = field(default_factory=lambda: np.eye(12))

class MSCKF:
    def __init__(self, parameters: MSCKFParameters, rr = None):
        self.rr = rr
                
        #* State
        self.state = MSCKFState() 
        self.state.imu = IMU(T_W_I=Isometry3D(np.eye(3), np.zeros(3)), T_W_Ii=Isometry3D(np.eye(3), np.zeros(3)), W_gravity=parameters.W_gravity)
        
        #* Camera
        self.K = parameters.K
        self.T_W_C = parameters.T_W_C
        self.width = parameters.width
        self.height = parameters.height
        self.sigma_image = parameters.sigma_image
                
        #* IMU
        self.only_imu = parameters.only_imu
        self.state.imu.W_gravity = parameters.W_gravity
        
        self.imu_measuremenets_buffer: List[IMUMeasurement] = []
        self.continuous_noise_covariance = np.eye(12)
        self.continuous_noise_covariance[:3, :3] = np.eye(3) * parameters.gyroscope_noise_density**2
        self.continuous_noise_covariance[3:6, 3:6] = np.eye(3) * parameters.gyroscope_random_walk**2
        self.continuous_noise_covariance[6:9, 6:9] = np.eye(3) * parameters.accelerometer_noise_density**2
        self.continuous_noise_covariance[9:12, 9:12] = np.eye(3) * parameters.accelerometer_random_walk**2
        self.state.continuous_noise_covariance = self.continuous_noise_covariance
                
        #* Features
        self.features: Dict[int, Feature] = {} # index, feature
        self.feature_extractor = FeatureExtractor()
        self.last_feature_index = 0
        self.first_measurement_arrived = False
        self.last_camera_measurement = None
        
        self.number_of_extracted_features = parameters.number_of_extracted_features
        self.use_parallax = parameters.use_parallax
        self.min_parallax = parameters.min_parallax
        self.min_cosine_similarity = parameters.min_cosine_similarity
        self.epipolar_rejection_threshold = parameters.epipolar_rejection_threshold
        self.homography_rejection_threshold = parameters.homography_rejection_threshold
        self.min_number_of_frames_to_be_lost = parameters.min_number_of_frames_to_be_lost if parameters.min_number_of_frames_to_be_lost > 1 else 1
        self.min_number_of_frames_to_be_tracked = parameters.min_number_of_frames_to_be_tracked if parameters.min_number_of_frames_to_be_tracked > 2 else 2 
        
        self.max_number_of_camera_states = parameters.max_number_of_camera_states
        self.camera_states_to_delete = int(self.max_number_of_camera_states/3)
        
        self.estimated_world_points = [] #! DEBUG
        self.currently_processed_world_points = [] #! DEBUG
        self.number_of_features_discarder_for_homography_test = 0 #! DEBUG
        self.number_of_features_discarded_for_epipolar_test = 0 #! DEBUG
        self.number_of_residuals_discarded_for_gasting_test = 0 #! DEBUG
        self.current_image = None
        self.stacked_image = None   
        self.composed_image = None

    
    def imu_callback(self, imu_measurement: IMUMeasurement): 
        if not self.first_measurement_arrived:
            self.imu_measuremenets_buffer.append(imu_measurement)
            return

        if not self.state.imu.is_initialized:        
            self.state.imu.initialize(imu_measurements=self.imu_measuremenets_buffer)
            for imu_meas in self.imu_measuremenets_buffer:
                self.process_imu(imu_meas)
        
        self.process_imu(imu_measurement)
        
    def feature_callback(self, image: cv2.Mat, extracted_features: ExtractedFeature = None):    
        self.current_image = image #! DEBUG    
        if not self.first_measurement_arrived: self.first_measurement_arrived = True
        if not self.state.imu.is_initialized: return

        if not self.only_imu:                
            self.state_augmentation()
            self.add_camera_measurements(image, extracted_features)
            self.process_features()    
            
            if len(self.state.cameras) > self.max_number_of_camera_states:
                self.prune_poorest_camera_states()   

    def process_imu(self, imu_measurement: IMUMeasurement):
        imu = self.state.imu
        dt = imu_measurement.timestamp - imu.timestamp
        imu.timestamp = imu_measurement.timestamp
        imu.id += 1
                    
        angular_velocity = imu_measurement.angular_velocity - imu.gyroscope_bias
        linear_acceleration = imu_measurement.linear_acceleration - imu.accelerometer_bias
        imu.integrate(linear_acceleration=linear_acceleration, angular_velocity=angular_velocity, dt=dt)

        #* Discrete transition and noise covariance matrices
        # State ordering: [δθ (0:3), δbg (3:6), δv (6:9), δba (9:12), δp (12:15)]
        
        #* Construct continuous-time state transition matrix F (15x15).
        # F Row 1 to 3: -skew(angular_velocity), -I_3x3, 0_3x3, 0_3x3, 0_3x3
        # F Row 4 to 6: 0_3x3, 0_3x3, 0_3x3, 0_3x3, 0_3x3
        # F Row 7 to 9: -R_W_I @ skew(linear_acceleration), 0_3x3, -2*skew(planet_angular_velocity), -R_W_I, -skew(planet_angular_velocity) @ -skew(planet_angular_velocity)
        # F Row 9 to 12: 0_3x3, 0_3x3, 0_3x3, 0_3x3, 0_3x3
        # F Row 12 to 15: 0_3x3, 0_3x3, I_3x3, 0_3x3, 0_3x3
        F = np.zeros((15, 15))
        
        # Row 1 to 3
        F[0:3, 0:3] = -skew(angular_velocity) 
        F[0:3, 3:6] = -np.eye(3)
        
        # Row 7 to 9
        F[6:9, 0:3] = -imu.T_W_Ii.R @ skew(linear_acceleration)
        F[6:9, 6:9] = -2*skew(self.state.imu.planet_angular_velocity)
        F[6:9, 9:12] = -imu.T_W_Ii.R
        F[6:9, 12:15] = -skew(self.state.imu.planet_angular_velocity) @ -skew(self.state.imu.planet_angular_velocity)
        
        # Row 12 to 15
        F[12:15, 6:9] = np.eye(3)
         
        #* Construct continuous-time noise input matrix G (15x12).
        # G Row 1 to 3:   -I_3x3, 0_3x3,  0_3x3, 0_3x3
        # G Row 4 to 6:    0_3x3, I_3x3,  0_3x3, 0_3x3
        # G Row 7 to 9:    0_3x3, 0_3x3, -R_W_I, 0_3x3
        # G Row 9 to 12:   0_3x3, 0_3x3,  0_3x3, I_3x3
        # G Row 12 to 15:  0_3x3, 0_3x3,  0_3x3, 0_3x3
        G = np.zeros((15, 12))

        # Row 1 to 3
        G[0:3, 0:3] = -np.eye(3)
        
        # Row 4 to 6
        G[3:6, 3:6] = np.eye(3)
        
        # Row 7 to 9
        G[6:9, 6:9] = -imu.T_W_Ii.R
        
        # Row 9 to 12
        G[9:12, 9:12] = np.eye(3)
        
        #* Phi transition matrix (truncated Taylor series expansion).
        Fdt = F * dt
        Fdt2 = Fdt @ Fdt
        Fdt3 = Fdt2 @ Fdt
        Phi = np.eye(15) + Fdt + 0.5* Fdt2 + (1.0/6.0)*Fdt3
        
        #* Enforce observability constraint
        Phi[:3, :3] = self.state.imu.T_W_Ii.R @ self.state.imu.T_W_Ii_null.R.T
        
        u = self.state.imu.T_W_Ii_null.R @ self.state.imu.W_gravity
        s = u / (u @ u)
        
        A_vel = Phi[6:9, :3].copy()
        A_pos = Phi[12:15, :3].copy()
        
        w1 = skew(self.state.imu.v_W_Ii_null - self.state.imu.v_W_Ii) @ self.state.imu.W_gravity
        w2 = skew(dt * self.state.imu.v_W_Ii_null + self.state.imu.T_W_Ii_null.t - self.state.imu.T_W_Ii.t) @ self.state.imu.W_gravity
        
        Phi[6:9, :3] = A_vel - (A_vel @ u - w1)[:, None] * s
        Phi[12:15, :3] = A_pos - (A_pos @ u - w2)[:, None] * s
        
        #* Update covariance
        P_imu = self.state.covariance[:15, :15] 
        Q = Phi @ G @ self.continuous_noise_covariance @ G.T @ Phi.T * dt
        self.state.covariance[:15, :15] = Phi @ P_imu @ Phi.T + Q
        
        P_imu_cam = self.state.covariance[:15, 15:]
        self.state.covariance[:15, 15:] = Phi @ P_imu_cam
        self.state.covariance[15:, :15] = self.state.covariance[:15, 15:].T
            
        self.state.covariance = (self.state.covariance + self.state.covariance.T) / 2
        
        #* Update null-state
        self.state.imu.T_W_Ii_null = self.state.imu.T_W_Ii
        self.state.imu.v_W_Ii_null = self.state.imu.v_W_Ii
        
    def state_augmentation(self):        
        #* Add new camera state
        T_I_C = self.state.imu.T_W_I.inv() * self.T_W_C   # static transformation (camera frame in IMU frame)  
        T_W_Ci = self.state.imu.T_W_Ii * T_I_C            # computed transformation (camera frame in global frame)
        camera = Camera(K=self.K, width=self.width, height=self.height, T_W_Ci=T_W_Ci)
        self.state.cameras[self.state.imu.id] = camera
        
        #* Reshape covariance       
        J = np.zeros((6, self.state.covariance.shape[0]))
        J[:3, :3] = T_I_C.R.T
        J[3:6,:3] = skew(self.state.imu.T_W_Ii.R @ T_I_C.t)
        J[3:6, 12:15] = np.eye(3)
        M = np.vstack((np.eye(self.state.covariance.shape[0]), J))

        state_covariance = M @ self.state.covariance @ M.T
        self.state.covariance = (state_covariance + state_covariance.T) / 2

    
    def add_camera_measurements(self, image: cv2.Mat, extracted_features: ExtractedFeature = None): 
        
        if extracted_features is None:  
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            output = self.feature_extractor.extract_features(grayscale_image, self.number_of_extracted_features)
            keypoints = output.keypoints
            descriptors = output.descriptors
            scores = output.scores
        else:
            keypoints = extracted_features.keypoints
            descriptors = extracted_features.descriptors
            scores = extracted_features.scores
            
        score_mean = np.mean(scores)
        keypoints = [keypoint for i, keypoint in enumerate(keypoints) if scores[i] >= 0.5*score_mean]
        descriptors = [descriptor for i, descriptor in enumerate(descriptors) if scores[i] >= 0.5*score_mean]
        scores = [score for score in scores if score >= 0.5*score_mean]
                
        if len(keypoints) == 0: return
        
        current_camera_id = self.state.imu.id
        camera = self.state.cameras[current_camera_id]        

        if len(self.features) == 0:
            for i, keypoint in enumerate(keypoints):
                self.last_feature_index += 1  
                                
                descriptor = descriptors[i]
                score = scores[i]
                Ci_v = camera.inverse_project_point(keypoint)
                W_v = camera.Ci2W(Ci_v, is_versor=True)            
                
                feature = Feature()
                feature.keypoints.append(keypoint)
                feature.descriptors.append(descriptor)
                feature.scores.append(score)
                feature.camera_indices.append(current_camera_id)
                feature.lines.append(Line(camera.T_W_Ci.t, W_v, score))
                feature.inverse_depth_point = InverseDepthPoint(camera.T_W_Ci, W_v)
                feature.tracked_for_n_frames += 1        
                feature.color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) #! DEBUG
                self.features[self.last_feature_index] = feature
            
            self.last_camera_measurement = CameraMeasurement(descriptors = np.array(descriptors), features_indices = np.array(list(self.features.keys())))
            
        
        else:               
            current_camera_measurement = CameraMeasurement(keypoints = np.array(keypoints), descriptors = np.array(descriptors), scores = np.array(scores))
            
            matched, not_matched, idxs = self.feature_extractor.match(self.last_camera_measurement, current_camera_measurement, self.min_cosine_similarity)
            lost_features_indices = not_matched.features_indices

            if len(matched.keypoints) == 0: return
            
            #! DEBUG
            if self.stacked_image is None: self.stacked_image = self.current_image
            else: self.stacked_image = cv2.addWeighted(self.stacked_image, 0.5, image, 0.5, 0)
            self.composed_image = np.hstack((self.stacked_image, self.current_image))
            debug_keypoints = []
            debug_keypoints_colors = []
            debug_keypoints_radius = []
            debug_lines = []
            debug_lines_colors = []
                        
            invK = np.linalg.inv(self.K)
            for i in range(len(matched.keypoints)): # Update matched features
                
                matched_keypoint = matched.keypoints[i]
                matched_descriptor = matched.descriptors[i]
                matched_score = matched.scores[i]
                matched_feature = matched.features_indices[i]
                feature = self.features[matched_feature]
                
                debug_keypoints.append([int(matched_keypoint[0] + self.width), int(matched_keypoint[1])])
                debug_keypoints_colors.append(feature.color)
                debug_keypoints_radius.append(3)
                
                epipolar_test_passed = True
                for j in range(len(feature.keypoints)):
                    feature_keypoint = feature.keypoints[j]
                    camera_index = feature.camera_indices[j]
                    feature_camera = self.state.cameras[camera_index]
                    T_C1_C2 = feature_camera.T_W_Ci.inv() * camera.T_W_Ci
                    
                    if np.linalg.norm(T_C1_C2.t) < 0.01:
                        H = self.K @ T_C1_C2.R @ invK
                        x1_pred = np.linalg.inv(H) @ np.array([matched_keypoint[0], matched_keypoint[1], 1])
                        x1_pred = x1_pred[:2] / x1_pred[2]
                        x2_pred = H @ np.array([feature_keypoint[0], feature_keypoint[1], 1])
                        x2_pred = x2_pred[:2] / x2_pred[2]
                        score = (np.linalg.norm(matched_keypoint - x1_pred) + np.linalg.norm(feature_keypoint - x2_pred)) / 2
                        
                        if score > self.homography_rejection_threshold: 
                            epipolar_test_passed = False
                            self.number_of_features_discarder_for_homography_test += 1
                            debug_keypoints.append([int(matched_keypoint[0] + self.width), int(matched_keypoint[1])])
                            debug_keypoints_colors.append([0, 0, 255])
                            debug_keypoints_radius.append(5)
                            if j == len(feature.keypoints) - 1: 
                                debug_lines.append([[int(matched_keypoint[0] + self.width), int(matched_keypoint[1])], [int(feature_keypoint[0]), int(feature_keypoint[1])]])
                                debug_lines_colors.append([0, 0, 255])
                            break
                        else:
                            debug_keypoints.append([int(feature_keypoint[0]), int(feature_keypoint[1])])
                            debug_keypoints_colors.append(feature.color)
                            debug_keypoints_radius.append(3)
                            if j == len(feature.keypoints) - 1: 
                                debug_lines.append([[int(matched_keypoint[0] + self.width), int(matched_keypoint[1])], [int(feature_keypoint[0]), int(feature_keypoint[1])]])  
                                debug_lines_colors.append([0, 255, 0])
                    else:
                        F = invK.T @ skew(T_C1_C2.t) @ T_C1_C2.R @ invK
                        score = np.append(matched_keypoint, 1).T @ F @ np.append(feature_keypoint, 1)
                        
                        if score > self.epipolar_rejection_threshold:
                            epipolar_test_passed = False
                            self.number_of_features_discarded_for_epipolar_test += 1
                            debug_keypoints.append([int(matched_keypoint[0] + self.width), int(matched_keypoint[1])])   
                            debug_keypoints_colors.append([255, 0, 0])
                            debug_keypoints_radius.append(5)
                            if j == len(feature.keypoints) - 1: 
                                debug_lines.append([[int(matched_keypoint[0] + self.width), int(matched_keypoint[1])], [int(feature_keypoint[0]), int(feature_keypoint[1])]])
                                debug_lines_colors.append([255, 0, 0])
                            break
                        else:
                            debug_keypoints.append([int(feature_keypoint[0]), int(feature_keypoint[1])])
                            debug_keypoints_colors.append(feature.color)
                            debug_keypoints_radius.append(3)
                            if j == len(feature.keypoints) - 1: 
                                debug_lines.append([[int(matched_keypoint[0] + self.width), int(matched_keypoint[1])], [int(feature_keypoint[0]), int(feature_keypoint[1])]])
                                debug_lines_colors.append([0, 255, 0])  
                    
                if not epipolar_test_passed: 
                    feature.lost_for_n_frames += 1
                    continue

                Ci_v = camera.inverse_project_point(matched_keypoint)
                W_v = camera.Ci2W(Ci_v, is_versor=True)      
                
                feature.keypoints.append(matched_keypoint)
                feature.descriptors.append(matched_descriptor)
                feature.scores.append(matched_score)
                feature.camera_indices.append(current_camera_id)
                feature.lines.append(Line(camera.T_W_Ci.t, W_v, matched_score))
                feature.tracked_for_n_frames += 1
                feature.lost_for_n_frames = 0
                        
            for i in range(len(not_matched.keypoints)): # Add new features
                
                not_matched_keypoint = not_matched.keypoints[i]
                not_matched_descriptor = not_matched.descriptors[i]
                not_matched_score = not_matched.scores[i]
                      
                self.last_feature_index += 1
                
                Ci_v = camera.inverse_project_point(not_matched_keypoint)
                W_v = camera.Ci2W(Ci_v, is_versor=True)
                
                feature = Feature()
                feature.keypoints.append(not_matched_keypoint)
                feature.descriptors.append(not_matched_descriptor)
                feature.scores.append(not_matched_score)
                feature.camera_indices.append(current_camera_id)
                feature.lines.append(Line(camera.T_W_Ci.t, W_v, not_matched_score))
                feature.inverse_depth_point = InverseDepthPoint(camera.T_W_Ci, W_v)
                feature.tracked_for_n_frames += 1
                feature.color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) #! DEBUG
                self.features[self.last_feature_index] = feature
            
            self.last_camera_measurement = CameraMeasurement()
            for index, feature in self.features.items():
                if index in lost_features_indices: feature.lost_for_n_frames += 1
                self.last_camera_measurement.descriptors.append(np.average(feature.descriptors, axis=0, weights=feature.scores))
                self.last_camera_measurement.features_indices.append(index)
            self.last_camera_measurement.descriptors = np.array(self.last_camera_measurement.descriptors)
            self.last_camera_measurement.features_indices = np.array(self.last_camera_measurement.features_indices)
            self.last_camera_measurement.keypoints = current_camera_measurement.keypoints
            self.last_camera_measurement.scores = current_camera_measurement.scores
            
            if self.rr is not None:
                rr.log('/camera_image/strips', rr.LineStrips2D(debug_lines, radii=[0.5]*len(debug_lines), colors=debug_lines_colors))
                rr.log('/camera_image/keypoints', rr.Points2D(debug_keypoints, radii=debug_keypoints_radius, colors=debug_keypoints_colors))
               
    def process_features(self):     
        self.currently_processed_world_points = [] #! DEBUG
           
        valid_features, lost_features = self.get_valid_features(self.features)     
        if len(valid_features) > 0:             
            self.update(valid_features)      
            self.remove_features(lost_features)
               
    def get_valid_features(self, features: Dict[int, Feature]) -> Tuple[Dict[int, Feature], Dict[int, Feature]]:
        valid_features: Dict[int, Feature] = {}
        lost_features: Dict[int, Feature] = {}
        for index, feature in features.items(): 
            
            lost = False
            if feature.lost_for_n_frames >= self.min_number_of_frames_to_be_lost:
                lost = True
            
            if lost and feature.tracked_for_n_frames < self.min_number_of_frames_to_be_tracked:
                lost_features[index] = feature
                continue
            
            enough_parallax = False
            if self.use_parallax and len(feature.lines) > 1:
                first_line = feature.lines[0]
                last_line = feature.lines[-1]
                parallax = np.rad2deg(angle_between_directions(first_line.direction, last_line.direction))
                if parallax > self.min_parallax:
                    enough_parallax = True
            
            if lost or enough_parallax:
                W_p_hat, C = intersection_of_lines(feature.lines)
                camera = self.state.cameras[feature.camera_indices[0]]
                Ci_p_hat = camera.W2Ci(W_p_hat)
                res, Im_p_hat = camera.project_point(Ci_p_hat)
                if res:
                    depth = Ci_p_hat[2]
                    Ci_v = camera.inverse_project_point(Im_p_hat)
                    W_v = camera.Ci2W(Ci_v, is_versor=True)
                    feature.inverse_depth_point.update(depth, W_v)
                    self.estimated_world_points.append(W_p_hat) #! DEBUG
                    self.currently_processed_world_points.append(W_p_hat) #! DEBUG
                    
                valid_features[index] = feature
                if lost: lost_features[index] = feature
            
        return valid_features, lost_features
            
    def compute_residual_and_jacobians(self, feature: Feature) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        observations = feature.keypoints
        camera_indices = feature.camera_indices 

        rj = []
        Hj_x = []
        Hj_f = []
        
        for i, camera_index in enumerate(camera_indices):
            camera = self.state.cameras[camera_index]  
            R_C_W = camera.T_W_Ci.R.T
            t_W_C = camera.T_W_Ci.t
            R_C_W_null = camera.T_W_Ci_null.R.T
            t_W_C_null = camera.T_W_Ci_null.t

            rho = feature.inverse_depth_point.rho
            base = feature.inverse_depth_point.base
            m = feature.inverse_depth_point.m

            Ci_f = R_C_W @ (rho * (base - t_W_C) + m)
            W_f = camera.Ci2W(Ci_f)

            z_i = np.linalg.inv(self.K) @ np.append(observations[i], 1)
            z_i = z_i[:2] / z_i[2]
            
            z_i_hat = [Ci_f[0] / Ci_f[2], Ci_f[1] / Ci_f[2]]
                        
            rj_i = (z_i - z_i_hat).reshape(2, 1)    
            
            Hj_xi_, Hj_fi_ = camera.compute_jacobians(Ci_f)
            
            u = np.zeros(6)
            u[:3] = R_C_W_null @ self.state.imu.W_gravity
            u[3:] = skew(W_f - t_W_C_null) @ self.state.imu.W_gravity
            
            A = Hj_xi_.copy()
            den = u @ u
            if den > 1e-6: A = A - (A @ u)[:, None] * u / den

            Hj_fi = -Hj_xi_[:, 3:]
            
            Hj_xi = np.zeros((2, self.state.covariance.shape[1]))
            index = list(self.state.cameras.keys()).index(camera_index) 
            Hj_xi[:, 15+index*6: 15+(index+1)*6] = A
                    
            rj.append(rj_i)
            Hj_x.append(Hj_xi)
            Hj_f.append(Hj_fi)          
                
        rj = np.vstack(rj)
        Hj_x = np.vstack(Hj_x)
        Hj_f = np.vstack(Hj_f)
                        
        rj_o, Hj_o = self.project_on_nullspace(Hj_f, rj, Hj_x) # rj_o: 2*Mj-3 x 1 ; Hj_o: 2*Mj-3 x 15+6N
            
        return rj_o, Hj_o
        
    def project_on_nullspace(self, Hj_f: np.ndarray, rj: np.ndarray, Hj_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        A = null_space(Hj_f.T) # left null space of Hj_f: 2*Mj x 2*Mj-rank(Hj_f)
        rj_o = A.T @ rj   # (2*Mj-3 x 2*Mj) x (2*Mj x 1)     = 2*Mj-3 x 1  
        Hj_o = A.T @ Hj_x # (2*Mj-3 x 2*Mj) x (2*Mj x 15+6N) = 2*Mj-3 x 15+6N
                
        return rj_o, Hj_o
    
    def gating_test(self, r: np.ndarray, H: np.ndarray) -> bool:        
        S_inv = np.linalg.inv(H @ self.state.covariance @ H.T + self.sigma_image**2 * np.eye(H.shape[0]))
        gamma = (r.T @ S_inv @ r).flatten()[0]
        dof = r.shape[0]
        alpha = 0.95
        critical_value = chi2.ppf(alpha, dof).flatten()[0]
            
        return gamma <= critical_value # True if the test is passed, False otherwise

    def update(self, features: Dict[int, Feature]):                       
        H_X_list = []  
        r_o_list = []
        for index, feature in features.items():
            rj_o, Hj_o = self.compute_residual_and_jacobians(feature)
            
            passed = self.gating_test(rj_o, Hj_o)
            if not passed: 
                self.number_of_residuals_discarded_for_gasting_test += 1
                continue
                        
            H_X_list.append(Hj_o)
            r_o_list.append(rj_o)
        
        if len(H_X_list) == 0: 
            return
        
        H_X = np.vstack(H_X_list)
        r_o = np.vstack(r_o_list) 
        R_o = self.sigma_image**2 * np.eye(r_o.shape[0])

        if H_X.shape[0] == 0 or H_X.shape[1] == 0 or r_o.shape[0] == 0: 
            return
        
        if H_X.shape[0]  > H_X.shape[1]:
            Q, R = np.linalg.qr(H_X, mode='reduced')
            T_H = R
            r_n = Q.T @ r_o
            R_n = Q.T @ R_o @ Q
        else:
            T_H = H_X
            r_n = r_o
            R_n = R_o
                    
        P = self.state.covariance
        S = (T_H @ P @ T_H.T) + R_n
        K = P @ T_H.T @ np.linalg.inv(S)
        delta_x = K @ r_n
                
        self.correct(K, T_H, R_n, delta_x)
        
    def correct(self, K: np.ndarray, T_H: np.ndarray, R_n: np.ndarray, delta_x: np.ndarray):   
        I = np.eye(self.state.covariance.shape[0])
        state_covariance = (I - K @ T_H) @ self.state.covariance @ (I - K @ T_H).T + K @ R_n @ K.T
        self.state.covariance = (state_covariance + state_covariance.T) / 2      

        delta_theta = delta_x[:3].flatten()                # Rotation correction from estimated IMU frame to true IMU frame (Theta_Ii_estIi) 
        delta_gyroscope_bias = delta_x[3:6].flatten()      # Gyroscope bias correction
        delta_velocity = delta_x[6:9].flatten()            # Velocity correction (IMU velocity)
        delta_accelerometer_bias = delta_x[9:12].flatten() # Accelerometer bias correction
        delta_position = delta_x[12:15].flatten()          # Position correction (IMU position)

        # delta_theta represents a small angle rotation vector from the estimated current IMU (estIi) frame to the true IMU frame (Ii)
        # The computation of the rotation matrix R_Ii_estiIi is done by the exponential map of delta_theta
        # The update of the current estimate of the IMU rotation is: R_W_Ii = R_W_estIi * R_Ii_estIi.T
        dThetaIMU_skew = skew(delta_theta)
        dThetaIMU_norm = np.linalg.norm(delta_theta)
        if np.isclose(dThetaIMU_norm, 0): R_Ii_estIi = np.eye(3)
        else: R_Ii_estIi = np.eye(3) + (np.sin(dThetaIMU_norm) / dThetaIMU_norm) * dThetaIMU_skew + ((1 - np.cos(dThetaIMU_norm)) / dThetaIMU_norm**2) * (dThetaIMU_skew @ dThetaIMU_skew)
        
        R_W_estIi = self.state.imu.T_W_Ii.R
        R_W_Ii = R_W_estIi @ R_Ii_estIi.T
        
        U, _, Vt = np.linalg.svd(R_W_Ii)
        R_W_Ii = U @ Vt
        self.state.imu.T_W_Ii.R = R_W_Ii
                
        self.state.imu.T_W_Ii.t += delta_position
        self.state.imu.v_W_Ii += delta_velocity
        self.state.imu.gyroscope_bias += delta_gyroscope_bias
        self.state.imu.accelerometer_bias += delta_accelerometer_bias

        # Correct all camera poses
        for i, (camera_id, camera) in enumerate(self.state.cameras.items()):
            delta_x_cam = delta_x[15+i*6: 21+i*6]
            delta_camera_theta = delta_x_cam[:3].flatten()
            delta_camera_position = delta_x_cam[3:6].flatten()
            
            # Correct camera
            dThetaCam_skew = skew(delta_camera_theta)
            dThetaCam_norm = np.linalg.norm(delta_camera_theta)
            if np.isclose(dThetaCam_norm, 0): R_Ci_estCi = np.eye(3)
            else: R_Ci_estCi = np.eye(3) + (np.sin(dThetaCam_norm) / dThetaCam_norm) * dThetaCam_skew + ((1 - np.cos(dThetaCam_norm)) / dThetaCam_norm**2) * (dThetaCam_skew @ dThetaCam_skew)
            
            R_W_estCi = camera.T_W_Ci.R
            R_W_Ci = R_W_estCi @ R_Ci_estCi.T
            
            U, _, Vt = np.linalg.svd(R_W_Ci)
            R_W_Ci = U @ Vt
            
            camera.T_W_Ci.R = R_W_Ci
            camera.T_W_Ci.t += delta_camera_position
            
    def prune_camera_states(self):        
        camera_states_to_process: Dict[int, Camera] = {}
        for i, (camera_index, camera) in enumerate(self.state.cameras.items()):
            if i>0 and i % int(self.max_number_of_camera_states/self.camera_states_to_delete) == 0:
                camera_states_to_process[camera_index] = camera
                
        features_to_process: Dict[int, Feature] = {}
        for index, feature in self.features.items():
            for camera_index in feature.camera_indices:
                if camera_index in camera_states_to_process.keys():
                    features_to_process[index] = feature
                    break
        
        valid_features, _ = self.get_valid_features(features_to_process)
        if len(valid_features) > 0: 
            self.update(valid_features)        
        
        self.remove_cameras(camera_states_to_process)
     
    # def prune_poorest_camera_states(self):        
    #     camera_states_to_process: Dict[int, Camera] = {}
        
    #     number_of_features_per_camera: Dict[int, int] = {} # camera_index, number_of_features
    #     for index, feature in self.features.items():
    #         for camera_index in feature.camera_indices:
    #             if camera_index in number_of_features_per_camera.keys(): number_of_features_per_camera[camera_index] += 1
    #             else: number_of_features_per_camera[camera_index] = 1
        
    #     sorted_number_of_features_per_camera = dict(sorted(number_of_features_per_camera.items(), key=lambda item: item[1]))
    #     for camera_index, _ in sorted_number_of_features_per_camera.items():
    #         if len(camera_states_to_process) == self.camera_states_to_delete: break
    #         if camera_index not in list(self.state.cameras.keys())[len(self.state.cameras)-self.camera_states_to_delete:]:  
    #             camera_states_to_process[camera_index] = self.state.cameras[camera_index]
        
    #     features_to_process: Dict[int, Feature] = {}
    #     for index, feature in self.features.items():
    #         for camera_index in feature.camera_indices:
    #             if camera_index in camera_states_to_process.keys():
    #                 features_to_process[index] = feature
    #                 break
        
    #     valid_features, _ = self.get_valid_features(features_to_process)
    #     if len(valid_features) > 0: 
    #         self.update(valid_features)        
            
    #     self.remove_cameras(camera_states_to_process)
    
    def prune_poorest_camera_states(self):
        
        number_of_features_per_camera: Dict[int, int] = {} # camera_index, number_of_features
        for index, feature in self.features.items():
            for camera_index in feature.camera_indices:
                if camera_index in number_of_features_per_camera.keys(): number_of_features_per_camera[camera_index] += 1
                else: number_of_features_per_camera[camera_index] = 1
        
        sorted_number_of_features_per_camera = dict(sorted(number_of_features_per_camera.items(), key=lambda item: item[1]))
                       
        # Select up to the first two cameras with the lowest number of features
        camera_states_to_process = {
            camera_index: self.state.cameras[camera_index]
            for i, camera_index in enumerate(sorted_number_of_features_per_camera.keys()) if i < 2
        }
        
        features_to_process: Dict[int, Feature] = {}
        for index, feature in self.features.items():
            for camera_index in feature.camera_indices:
                if camera_index in camera_states_to_process.keys():
                    features_to_process[index] = feature
                    break
        
        valid_features, _ = self.get_valid_features(features_to_process)
        if len(valid_features) > 0: 
            self.update(valid_features)        
            
        self.remove_cameras(camera_states_to_process)
     
    def remove_features(self, features: Dict[int, Feature]):
        for index_feature in features.keys():
            del self.features[index_feature]
            if self.last_camera_measurement is not None:
                index = np.where(self.last_camera_measurement.features_indices == index_feature)[0]
                if len(index) > 0:
                    index = index[0]
                    self.last_camera_measurement.descriptors = np.delete(self.last_camera_measurement.descriptors, index, axis=0)
                    self.last_camera_measurement.features_indices = np.delete(self.last_camera_measurement.features_indices, index, axis=0)
            
        self.remove_cameras(self.get_cameras_without_features())
            
    def remove_cameras(self, cameras: Dict[int, Camera]):
        for index_camera in cameras.keys():
            index = list(self.state.cameras.keys()).index(index_camera)
            state_covariance = self.state.covariance
            state_covariance = np.delete(state_covariance, slice(15+index*6, 15+(index+1)*6), axis=0)
            state_covariance = np.delete(state_covariance, slice(15+index*6, 15+(index+1)*6), axis=1)
            self.state.covariance = state_covariance
            del self.state.cameras[index_camera]
        
        features_to_delete: Dict[int, Feature] = {}
        for index, feature in self.features.items():
            for index_camera in cameras.keys():
                if index_camera in feature.camera_indices:
                    camera_index = feature.camera_indices.index(index_camera)
                    del feature.keypoints[camera_index]
                    del feature.descriptors[camera_index]
                    del feature.scores[camera_index]
                    del feature.camera_indices[camera_index]
                    del feature.lines[camera_index]
            if len(feature.camera_indices) == 0: features_to_delete[index] = feature
        
        for index_feature in features_to_delete.keys():
            del self.features[index_feature]
            if self.last_camera_measurement is not None:
                index = np.where(self.last_camera_measurement.features_indices == index_feature)[0]
                if len(index) > 0:
                    index = index[0]
                    self.last_camera_measurement.descriptors = np.delete(self.last_camera_measurement.descriptors, index, axis=0)
                    self.last_camera_measurement.features_indices = np.delete(self.last_camera_measurement.features_indices, index, axis=0)
                    
    def get_cameras_without_features(self) -> Dict[int, Camera]:
        cameras_without_features: Dict[int, Camera] = {}   
        for index_camera, camera in self.state.cameras.items():
            camera_is_tracking = False
            for _, feature in self.features.items():
                if index_camera in feature.camera_indices:
                    camera_is_tracking = True
                    break
            if not camera_is_tracking: cameras_without_features[index_camera] = camera
        
        return cameras_without_features
    