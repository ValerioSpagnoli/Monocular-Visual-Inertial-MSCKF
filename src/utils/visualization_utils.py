import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from matplotlib.patches import Ellipse
import plotly.graph_objs as go

from src.utils.geometry import *

class CameraImage():
    def __init__(self, width: int, height: int, name: str = 'Camera', ax: np.ndarray=None) -> None:
        """
        Initializes the visualization utility with the specified width, height, and name.
        
        Args:
            width (int): The width of the image.
            height (int): The height of the image.
            name (str, optional): The name of the camera. Defaults to 'Camera'.
            ax (np.ndarray, optional): The axes object for plotting. Defaults to None.
        
        Attributes:
            width (int): The width of the image.
            height (int): The height of the image.
            image (np.ndarray): A numpy array representing the image, initialized to ones.
            name (str): The name of the camera.
            ax (matplotlib.axes.Axes): The axes object for plotting.
        """
        
        self.width = width
        self.height = height
        self.image = np.ones((height, width, 3))
        self.name = name
        
        if ax is None:
            fig, self.ax = plt.subplots()
        else:
            self.ax = ax
        
    def show(self) -> None:
        """
        Displays the image using matplotlib.

        This method shows the image stored in the instance using the `imshow` 
        function from matplotlib. It also sets the x and y labels, the title 
        of the plot, and adds a legend.

        Returns:
            None
        """
        self.ax.imshow(self.image)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title(self.name)
        self.ax.grid()
        self.ax.legend()
        for text in self.ax.get_legend().get_texts():
            text.set_fontsize('small')
            
    def save(self, filename: str) -> None:
        """
        Saves the image to a file.

        This method saves the image stored in the instance to a file with the
        specified filename.

        Args:
            filename (str): The name of the file to save the image to.

        Returns:
            None
        """
        self.ax.imshow(self.image)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title(self.name)
        self.ax.grid()
        self.ax.legend()
        for text in self.ax.get_legend().get_texts():
            text.set_fontsize('small')
        plt.savefig(filename)
            
    def plot_point(self, point: np.ndarray, color: str = 'red', shape: str = 'o', size: int = 5, label: str = 'Point') -> None:
        """
        Plots a single point on the current axes.

        Args:
            point (np.ndarray): A numpy array representing the point to plot.
            color (str, optional): The color of the point. Defaults to 'red'.
            shape (str, optional): The shape of the point. Defaults to 'o'.
            size (int, optional): The size of the point. Defaults to 5.
            label (str, optional): The label for the point, used in the legend. Defaults to 'Point'.

        Returns:
            None
        """
        self.ax.scatter(point[0], point[1], c=color, marker=shape, s=size, label=label)
        self.ax.legend()
        
    def plot_points(self, points: np.ndarray, color: str = 'red', shape: str = 'o', size: int = 5, label: str = 'Point') -> None:
        """
        Plots a single points on the current axes.

        Parameters:
            points : np.ndarray
                A numpy (2,N) array representing the points to plot.
                The x-coordinates are in the first row, and the y-coordinates are in the second row.
            color : str, optional
                The color of the points (default is 'red').
            label : str, optional
                The label for the points, used in the legend (default is 'Point').

        Returns:
            None
        """
        if(len(points) == 0): return
        self.ax.scatter(points[:, 0], points[:, 1], c=color, marker=shape, s=size, label=label)
        self.ax.legend()
        
    def plot_line(self, point1: np.ndarray, point2: np.ndarray, color: str = 'blue', label:str = 'Point') -> None:
        """
        Plots a line between two points on the current axes.

        Args:
            point1 (np.ndarray): The starting point of the line as a numpy array [x, y].
            point2 (np.ndarray): The ending point of the line as a numpy array [x, y].
            color (str, optional): The color of the line. Defaults to 'blue'.
            label (str, optional): The label for the line. Defaults to 'Point'.

        Returns:
            None
        """
        self.ax.plot([point1[0], point2[0]], [point1[1], point2[1]], color=color, label=label)
        self.ax.legend()
        
    def plot_gaussian(self, mean: np.ndarray, covariance: np.ndarray, sigma_bound: int = 3, color: str = 'blue') -> None:
        """
        Plots a Gaussian distribution as an ellipse on a 2D plot.

        Parameters:
            mean (np.ndarray): The mean of the Gaussian distribution, a 2D point.
            covariance (np.ndarray): The 2x2 covariance matrix of the Gaussian distribution.
            sigma_bound (int, optional): The number of standard deviations to determine the ellipse size. Default is 3.
            color (str, optional): The color of the ellipse and mean point. Default is 'blue'.

        Returns:
            None
        """
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * sigma_bound * np.sqrt(eigenvalues)
        ellipsoid = Ellipse(xy=(mean[0], mean[1]), width=width, height=height, angle=angle, edgecolor=color, fc='None', lw=1, label=f'{sigma_bound}-sigma bound')
        self.ax.add_patch(ellipsoid)
        self.ax.scatter(mean[0], mean[1], color=color, s=5)
                              
                
class Camera3D():
    def __init__(self, 
                width: int, 
                height: int,
                K: np.ndarray,
                T_C_Ci: Isometry3D,
                T_B_C: Isometry3D,
                image_plane_depth: float = 1, 
                name: str = 'Camera') -> None:
         
        self.width = width
        self.height = height
        self.K = K
        self.T_C_Ci = T_C_Ci
        self.T_B_C = T_B_C
        self.image_plane_depth = image_plane_depth
        self.name = name
        
        self.T_B_Ci = self.T_B_C * self.T_C_Ci
        
        fov_width = self.width / self.K[0, 0]    
        fov_height = self.height / self.K[1, 1]
                
        self.camera_plane = np.array([
            [-fov_width / 2, -fov_height / 2, image_plane_depth],  # Bottom-left
            [ fov_width / 2, -fov_height / 2, image_plane_depth],  # Bottom-right
            [ fov_width / 2,  fov_height / 2, image_plane_depth],  # Top-right
            [-fov_width / 2,  fov_height / 2, image_plane_depth]   # Top-left
        ])  
        
    def show(self, T_C_Ci: Isometry3D,  color: str = 'red') -> None:
        """
        Visualizes the camera position and field of view (FOV) in a 3D plot.
        Parameters:
            T_C_Ci (np.ndarray): A 4x4 transformation matrix representing the pose of the camera in the world coordinate frame.
            color (str, optional): The color of the camera position marker, by default 'red'.
        Returns:
            None
        """
        
        self.T_B_Ci = self.T_B_C * T_C_Ci
        self.B_c = self.T_B_Ci.t
        
        # self.camera_plane_in_C = np.array([(T_C_Ci @ np.append(point, 1))[:3] for point in self.camera_plane])
        self.camera_plane_in_C = np.array([(T_C_Ci.transform(point)) for point in self.camera_plane])
        # self.camera_plane_in_B = np.array([(self.T_B_C @ np.append(point, 1))[:3] for point in self.camera_plane_in_C])
        self.camera_plane_in_B = np.array([(self.T_B_C.transform(point)) for point in self.camera_plane_in_C])
        
        #* Camera position
        self.fig.add_trace(go.Scatter3d(
            x=[self.B_c[0]], y=[self.B_c[1]], z=[self.B_c[2]],
            mode='markers', marker=dict(size=3, color=color),
            name=self.name
        ))

        #* FOV lines
        for point in self.camera_plane_in_B:
            self.fig.add_trace(go.Scatter3d(
                x=[self.B_c[0], point[0]], y=[self.B_c[1], point[1]], z=[self.B_c[2], point[2]],
                mode='lines', line=dict(color='gray', width=2), showlegend=False,
            ))
        for i in range(4):
            self.fig.add_trace(go.Scatter3d(
                x=[self.camera_plane_in_B[i, 0], self.camera_plane_in_B[(i + 1) % 4, 0]],
                y=[self.camera_plane_in_B[i, 1], self.camera_plane_in_B[(i + 1) % 4, 1]],
                z=[self.camera_plane_in_B[i, 2], self.camera_plane_in_B[(i + 1) % 4, 2]],
                mode='lines', line=dict(color='gray', width=2), showlegend=False,
            ))

        #* Camera plane mesh
        self.fig.add_trace(go.Mesh3d(
            x=self.camera_plane_in_B[:, 0], y=self.camera_plane_in_B[:, 1], z=self.camera_plane_in_B[:, 2],
            i=[0, 0, 1, 1], j=[1, 2, 2, 3], k=[2, 3, 3, 0],
            opacity=0.2, color='lightblue',
            name='Camera Plane'
        ))
                
class Canvas3D():
    def __init__(self,
                x_range: list = [-20, 20],
                y_range: list = [-20, 20],
                z_range: list = [-20, 20],
                title: str = 'World') -> None:
        """
        Initializes the visualization utility with specified ranges and title.
        Args:
            x_range (list, optional): The range for the x-axis. Defaults to [-20, 20].
            y_range (list, optional): The range for the y-axis. Defaults to [-20, 20].
            z_range (list, optional): The range for the z-axis. Defaults to [-20, 20].
            title (str, optional): The title of the plot. Defaults to 'World'.
        Returns:
            None
        """
        
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.title = title
        
        self.fig = go.Figure()
        self.fig.update_layout(
            title=self.title,
            scene=dict(
                xaxis=dict(title='X', range=self.x_range),
                yaxis=dict(title='Y', range=self.y_range),
                zaxis=dict(title='Z', range=self.z_range),
                aspectmode='manual',
                aspectratio=dict(x=(self.x_range[1] - self.x_range[0]) / (self.x_range[1] - self.x_range[0]),
                                y=(self.y_range[1] - self.y_range[0]) / (self.x_range[1] - self.x_range[0]),
                                z=(self.z_range[1] - self.z_range[0]) / (self.x_range[1] - self.x_range[0]))
                                    
                
            ),
            scene_camera=dict(
                eye=dict(x=-0.75, y=-0.75, z=0.75)                
            ),
            width=1200,
            height=800,
            showlegend=True
        )    
        
    def show(self) -> None:
        """
        Displays the figure.

        This method will render and display the figure associated with the instance.
        """
        self.fig.show()
        
    def add_camera(self, camera: Camera3D, T_C_Ci: np.ndarray=None, color: str='red') -> None:
        """
        Adds a 3D camera to the visualization.

        Parameters:
        camera (Camera3D): The 3D camera object to be added.
        T_C_Ci (np.ndarray, optional): The transformation matrix from world coordinates to camera coordinates. 
                                       If None, the camera's default transformation matrix is used. Default is None.
        color (str, optional): The color to use for the camera visualization. Default is 'red'.

        Returns:
        None
        """
        camera.fig = self.fig
        camera.show(T_C_Ci if T_C_Ci is not None else camera.T_C_Ci, color=color)
        
    def add_point(self, point: np.ndarray, color: str='black', size: int=5, name: str='Point') -> None:
        """
        Adds a 3D point to the figure.

        Parameters:
        point (np.ndarray): A numpy array containing the x, y, z coordinates of the point.
        color (str, optional): The color of the point. Default is 'black'.
        size (int, optional): The size of the point marker. Default is 5.
        name (str, optional): The name of the point trace. Default is 'Point'.

        Returns:
        None
        """
        self.fig.add_trace(go.Scatter3d(
            x=[point[0]], y=[point[1]], z=[point[2]],
            mode='markers', marker=dict(size=size, color=color),
            name=name
        ))
        
    def add_points(self, points: np.ndarray, color: str='black', size: int=5, name: str='Points') -> None:
        """
        Adds multiple 3D points to the figure.

        Parameters:
            points (np.ndarray): A numpy array containing the x, y, z coordinates of the points.
            color (str, optional): The color of the points. Default is 'black'.
            size (int, optional): The size of the point markers. Default is 5.
            name (str, optional): The name of the points trace. Default is 'Points'.

        Returns:
            None
        """
        self.fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='markers', marker=dict(size=size, color=color),
            name=name
        ))
        
    def add_poses(self, poses: np.ndarray, color: str='black', size: int=5, name: str='Poses') -> None:
        points = []
        lines = []
        for pose in poses:
            position = pose[:3]
            orientation = pose[3:]
            points.append(position)
            Rz_ = Rz(orientation[2])
            Ry_ = Ry(orientation[1])
            Rx_ = Rx(orientation[0])
            R = Rz_ @ Ry_ @ Rx_
            direction_local = np.array([1, 0, 0])
            direction = R @ direction_local
            
            lines.append((position, position + direction*0.5))
        
        # Add all points with a single name
        for point in points:
            self.add_point(point, color=color, size=size, name=name)
        
        # Add all lines with a single name
        for line in lines:
            self.add_line(line[0], line[1], color=color, width=size, name=name)
        
    def add_line(self, point1: np.ndarray, point2: np.ndarray, color: str='black', width: int=1, name: str='Line', show_legend: bool=False) -> None:
        """
        Adds a 3D line to the figure.

        Parameters:
            point1 (np.ndarray): A 3-element array representing the starting point of the line (x, y, z).
            point2 (np.ndarray): A 3-element array representing the ending point of the line (x, y, z).
            color (str, optional): The color of the line. Default is 'black'.
            width (int, optional): The width of the line. Default is 1.
            name (str, optional): The name of the line trace. Default is 'Line'.
            show_legend (bool, optional): Whether to show the line in the legend. Default is False.

        Returns:
            None
        """

        self.fig.add_trace(go.Scatter3d(
            x=[point1[0], point2[0]], y=[point1[1], point2[1]], z=[point1[2], point2[2]],
            mode='lines', line=dict(color=color, width=width),
            name=name, showlegend=show_legend
        ))
        
    def add_ellipsoide(self, center: np.ndarray, covariance: np.ndarray, color: str='blue', name: str='Ellipsoide', scale: float=1.0) -> None:
        """
        Adds a 3D ellipsoid to the figure.
    
        Parameters:
            center (np.ndarray): A 3-element array representing the center of the ellipsoid (x, y, z).
            covariance (np.ndarray): A 3x3 covariance matrix representing the shape of the ellipsoid.
            color (str, optional): The color of the ellipsoid. Default is 'blue'.
            name (str, optional): The name of the ellipsoid trace. Default is 'Ellipsoide'.
            scale (float, optional): A scale factor for the ellipsoid. Default is 1.0.
    
        Returns:
            None
        """
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        radii = np.sqrt(eigenvalues) * scale
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = radii[0] * np.cos(u) * np.sin(v)
        y = radii[1] * np.sin(u) * np.sin(v)
        z = radii[2] * np.cos(v)
        for i in range(len(x)):
            for j in range(len(x[i])):
                [x[i][j], y[i][j], z[i][j]] = np.dot([x[i][j], y[i][j], z[i][j]], eigenvectors) + center
        self.fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, color], [1, color]], name=name, showscale=False))