from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import cv2
import torch
import numpy as np

from src.msckf.Camera import Camera 
from src.utils.geometry import *

# XFeat imports its internals as `modules.*`: src/accelerated_features must be on PYTHONPATH (set in pixi.toml)
from modules.xfeat import XFeat

@dataclass
class Feature:
    keypoints: List[np.ndarray] = field(default_factory=list)     # 2D positions of the feature in the image in each frame in which it was observed
    descriptors: List[np.ndarray] = field(default_factory=list)   # Descriptors of the feature in each frame in which it was observed
    scores: List[np.ndarray] = field(default_factory=list)        # Score of the feature extraction procedure
    camera_indices: List[int] = field(default_factory=list)       # Indices of frames in which the feature was observed
    lines: List[Line] = field(default_factory=list)               # Inverse projection of each keypoint
    inverse_depth_point: InverseDepthPoint = field(default_factory=InverseDepthPoint)  # Inverse depth point of the feature
    world_point: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 3D position of the feature in the world frame
    tracked_for_n_frames: int = 0                                 # Number of frames in which the feature was observed
    lost_for_n_frames: int = 0                                    # Number of frames in which the feature was not observed

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

def resolve_device(device: str = 'auto') -> torch.device:
    """Return the requested torch device; 'auto' picks cuda, then mps, then cpu."""
    if device == 'auto':
        if torch.cuda.is_available(): return torch.device('cuda')
        if torch.backends.mps.is_available(): return torch.device('mps')
        return torch.device('cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, falling back to CPU')
        return torch.device('cpu')
    if device == 'mps' and not torch.backends.mps.is_available():
        print('MPS not available, falling back to CPU')
        return torch.device('cpu')
    return torch.device(device)

class FeatureExtractor():
    def __init__(self, device: str = 'auto'):
        self.xfeat = XFeat()
        
        # XFeat always selects cuda/cpu by itself: override its device after loading the weights
        self.device = resolve_device(device)
        self.xfeat.dev = self.device
        self.xfeat.net = self.xfeat.net.to(self.device)
        print(f'XFeat running on: {self.device}')
        
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
