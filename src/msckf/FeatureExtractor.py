from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import cv2
import torch
import numpy as np

from src.msckf.Camera import Camera 
from src.utils.geometry import *
from src.accelerated_features.modules.xfeat import XFeat

@dataclass
class Feature:
    keypoints: List[np.ndarray] = field(default_factory=list)     # 2D positions of the feature in the image in each frame in which it was observed
    descriptors: List[np.ndarray] = field(default_factory=list)   # Descriptors of the feature in each frame in which it was observed
    scores: List[np.ndarray] = field(default_factory=list)        # Score of the feature extraction procedure
    camera_indices: List[int] = field(default_factory=list)       # Indices of frames in which the feature was observed
    lines: List[Line] = field(default_factory=list)               # Inverse projection of each keypoint
    inverse_depth_point: InverseDepthPoint = InverseDepthPoint()  # Inverse depth point of the feature
    world_point: np.ndarray = np.zeros(3)                         # 3D position of the feature in the world frame
    tracked_for_n_frames: int = 0                                 # Number of frames in which the feature was observed
    lost_for_n_frames: int = 0                                    # Number of frames in which the feature was not observed
    color: np.ndarray = np.zeros(3)                               #! DEBUG

@dataclass
class CameraMeasurement:
    keypoints: List[np.ndarray] = field(default_factory=list)
    descriptors: List[np.ndarray] = field(default_factory=list)
    scores: List[np.ndarray] = field(default_factory=list)
    features_indices: List[int] = field(default_factory=list)

@dataclass
class ExtractedFeature:
    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: np.ndarray

class FeatureExtractor():
    def __init__(self):
        self.xfeat = XFeat()
        
    def extract_features(self, image: np.ndarray, top_k: int = 256) -> ExtractedFeature:
        output = self.xfeat.detectAndCompute(image, top_k=top_k)[0]
        keypoints = output['keypoints'].squeeze(0)
        descriptors = output['descriptors'].squeeze(0)
        scores = output['scores']
                
        output = ExtractedFeature(keypoints=keypoints.cpu().numpy(),
                                  descriptors=descriptors.cpu().numpy(),
                                  scores=scores.cpu().numpy())
        
        return output
    
    def match(self, input1: CameraMeasurement, input2: CameraMeasurement, min_cosine_similarity: float = 0.82) -> Tuple[CameraMeasurement, CameraMeasurement, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        
        descriptors1_input = input1.descriptors
        descriptors2_input = input2.descriptors

        descriptors1_torch = torch.from_numpy(descriptors1_input)
        descriptors2_torch = torch.from_numpy(descriptors2_input)
        
        idxs1, idxs2 = self.xfeat.match(descriptors1_torch, descriptors2_torch, min_cossim=min_cosine_similarity)
        not_matched_idxs1 = np.setdiff1d(np.arange(len(descriptors1_input)), idxs1)
        not_matched_idxs2 = np.setdiff1d(np.arange(len(descriptors2_input)), idxs2)
                                                
        matched = CameraMeasurement(keypoints=np.atleast_2d(input2.keypoints[idxs2]),
                                    descriptors=np.atleast_2d(input2.descriptors[idxs2]),
                                    scores=np.atleast_1d(input2.scores[idxs2]),
                                    features_indices=np.atleast_1d(input1.features_indices[idxs1]))
        
        not_matched = CameraMeasurement(keypoints=np.atleast_2d(input2.keypoints[not_matched_idxs2]),
                                        descriptors=np.atleast_2d(input2.descriptors[not_matched_idxs2]),
                                        scores=np.atleast_1d(input2.scores[not_matched_idxs2]),
                                        features_indices=np.atleast_1d(input1.features_indices[not_matched_idxs1]))
        
        return matched, not_matched, (idxs1, idxs2, not_matched_idxs1, not_matched_idxs2)
    
    def match_frames(self, input1: CameraMeasurement, input2: CameraMeasurement) -> Tuple[CameraMeasurement, CameraMeasurement]:
        
        keypoints1_input = input1.keypoints
        descriptors1_input = input1.descriptors 
        scores1_input = input1.scores
        
        keypoints2_input = input2.keypoints
        descriptors2_input = input2.descriptors
        scores2_input = input2.scores 

        descriptors1_torch = torch.from_numpy(descriptors1_input)
        descriptors2_torch = torch.from_numpy(descriptors2_input)  
        
        idxs1, idxs2 = self.xfeat.match(descriptors1_torch, descriptors2_torch)
        
        keypoints1_output = keypoints1_input[idxs1]
        descriptors1_output = descriptors1_input[idxs1]
        scores1_output = scores1_input[idxs1]
        
        keypoints2_output = keypoints2_input[idxs2]
        descriptors2_output = descriptors1_input[idxs2]
        scores2_output = scores2_input[idxs2]
        
        output1 = CameraMeasurement(keypoints=keypoints1_output,
                                    descriptors=descriptors1_output,
                                    scores=scores1_output)
        
        output2 = CameraMeasurement(keypoints=keypoints2_output,
                                    descriptors=descriptors2_output,
                                    scores=scores2_output)
                
        return output1, output2
    
    def draw_matches(self, image1: np.ndarray, keypoints1: np.ndarray, image2: np.ndarray, keypoints2: np.ndarray):
        H, mask = cv2.findHomography(keypoints1, keypoints2, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
        mask = mask.flatten()

        h, w = image1.shape[:2]
        corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

        warped_corners = cv2.perspectiveTransform(corners_img1, H)

        img2_with_corners = image2.copy()
        for i in range(len(warped_corners)):
            start_point = tuple(warped_corners[i-1][0].astype(int))
            end_point = tuple(warped_corners[i][0].astype(int))
            cv2.line(img2_with_corners, start_point, end_point, (0, 0, 255), 4)

        keypoints1_cv2 = [cv2.KeyPoint(p[0], p[1], 5) for p in keypoints1]
        keypoints2_cv2 = [cv2.KeyPoint(p[0], p[1], 5) for p in keypoints2]
        matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

        img_matches = cv2.drawMatches(image1, keypoints1_cv2, img2_with_corners, keypoints2_cv2, matches, None, matchColor=(0, 255, 0), flags=2)

        return img_matches