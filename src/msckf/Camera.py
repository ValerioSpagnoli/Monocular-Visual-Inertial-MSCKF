from typing import Tuple
import numpy as np
from src.utils.geometry import *

class Camera:
    def __init__(self, K: np.ndarray, width: int, height: int, T_W_Ci: Isometry3D) -> None:
        self.K =  K
        self.width = width
        self.height = height
        self.T_W_Ci = T_W_Ci
        self.T_W_Ci_null = T_W_Ci
        
    def project_point(self, Ci_p: np.ndarray) -> Tuple[bool, np.ndarray]:
        #* Ci_p:  3D point in i-th Camera Frame coordinates
        #* Im_p:  2D point in image coordinates

        #* point is behind the camera
        if Ci_p[2] <= 0: return False, None
        
        Im_p = self.K @ Ci_p   
        Im_p = Im_p[:2] / Im_p[2]
        
        #* point is outside the camera plane
        if Im_p[0] < 0 or Im_p[0] >= self.width or \
           Im_p[1] < 0 or Im_p[1] >= self.height: 
            return False, None
        
        return True, Im_p
        
    def inverse_project_point(self, Im_p: np.ndarray) -> np.ndarray:
        #* Im_p: 2D image point coordinates
        #* Ci_v:  versor of the 3D point in camera coordinates
        #* Inverse Projection: Ci_v = inv(K) * [Im_p, 1]      
          
        Ci_v = np.linalg.inv(self.K) @ np.append(Im_p, 1)
        return Ci_v  
    
    def Ci2W(self, Ci_n: np.ndarray, is_versor: bool = False) -> np.ndarray:        
        #* T_W_Ci: Camera Frame in Camera World Frame 
        #* Ci_n:   Input 3D point/versor in Camera Frame
        #* W_n:    Output 3D point in World Frame

        W_n = self.T_W_Ci.transform(Ci_n, rotation_only=True if is_versor else False)
        return W_n
    
    def W2Ci(self, W_n: np.ndarray, is_versor: bool = False) -> np.ndarray:       
        #* T_W_Ci: Camera Frame in World Frame
        #* W_n:    Input 3D point/versor in World Frame
        #* Ci_n:   Output 3D point in Camera Frame

        Ci_n = self.T_W_Ci.inv().transform(W_n, rotation_only=True if is_versor else False)
        return Ci_n
   
    def compute_jacobians(self, Ci_f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Ci_x, Ci_y, Ci_z = Ci_f

        J_i = np.array([[1.0/Ci_z,       0, -Ci_x/(Ci_z**2)],
                        [     0,  1.0/Ci_z, -Ci_y/(Ci_z**2)]])
        
        R_Ci_W = self.T_W_Ci.R.T
        
        H_f = J_i @ R_Ci_W 
        
        H_x = np.zeros((2, 6))
        H_x[:, :3] = J_i @ skew(Ci_f)
        H_x[:, 3:] = - J_i @ R_Ci_W
        
        return H_x, H_f