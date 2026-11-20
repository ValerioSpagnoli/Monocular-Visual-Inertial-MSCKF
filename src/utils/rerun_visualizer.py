from typing import Dict, List, Optional
import numpy as np
import cv2
import rerun as rr
import rerun.blueprint as rrb

from src.utils.geometry import Isometry3D

#* Colors (RGB)
COLOR_GT = [40, 190, 80]           # ground truth
COLOR_EST = [40, 130, 255]         # estimate
COLOR_TRACK = [40, 220, 120]       # feature tracked across frames
COLOR_NEW = [80, 200, 255]         # feature detected in this frame
COLOR_UPDATE = [255, 160, 0]       # feature used in this frame's EKF update
COLOR_GATED = [200, 0, 200]        # feature discarded by the chi-square gating test
COLOR_REJECTED = [255, 50, 50]     # match discarded by the epipolar/homography test
COLOR_MAP = [200, 200, 200]        # triangulated landmarks
COLOR_WINDOW = [255, 200, 60]      # camera poses in the sliding window
COLOR_COVARIANCE = [255, 80, 80]

# Entity paths
CAMERA = 'world/est/body/camera'
METRICS = 'text/metrics'
STATUS = 'text/status'


def blueprint() -> rrb.Blueprint:
    """3D map on the left; camera image with features and textual metrics/status on the right."""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin='world', name='Map', background=[18, 18, 22], line_grid=rrb.LineGrid3D(visible=True)),
            rrb.Vertical(
                rrb.Spatial2DView(origin=CAMERA, name='Camera'),
                rrb.Horizontal(
                    rrb.TextDocumentView(origin=METRICS, name='Metrics'),
                    rrb.TextDocumentView(origin=STATUS, name='Filter status'),
                ),
                row_shares=[3, 2],
            ),
            column_shares=[3, 2],
        ),
        rrb.BlueprintPanel(state='collapsed'),
        rrb.SelectionPanel(state='collapsed'),
        rrb.TimePanel(state='collapsed'),
    )


class RerunVisualizer:
    """
    Logs the MSCKF state to Rerun.

    Fast-changing data (body poses) is logged at every IMU step; image, features, map and trajectories
    only at camera frames. Trajectories and map are re-logged as a whole every `history_every` frames,
    because Rerun shows only the latest value of an entity.
    """

    def __init__(self, K: np.ndarray, width: int, height: int, T_I_C: Isometry3D,
                 save_path: Optional[str] = None, track_length: int = 10, history_every: int = 5, jpeg_quality: int = 75, max_landmark_distance: float = 30.0):
        self.track_length = track_length
        self.history_every = history_every
        self.jpeg_quality = jpeg_quality
        self.max_landmark_distance = max_landmark_distance
        self.frame_count = 0

        self.est_trajectory: List[np.ndarray] = []
        self.gt_trajectory: List[np.ndarray] = []

        rr.init('vio')
        if save_path is not None:
            # Stream to the viewer and to the .rrd file at the same time (save/spawn alone would replace each other's sink)
            rr.spawn(connect=False)
            rr.set_sinks(rr.GrpcSink(), rr.FileSink(save_path))
        else:
            rr.spawn()
        rr.send_blueprint(blueprint())

        #* Static scene
        rr.log('world', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log('world/est/body', rr.TransformAxes3D(0.3), static=True)
        rr.log('world/gt/body', rr.TransformAxes3D(0.3), static=True)
        rr.log(CAMERA, rr.Transform3D(translation=T_I_C.t, mat3x3=T_I_C.R), static=True)
        rr.log(CAMERA, rr.Pinhole(image_from_camera=K, resolution=[width, height], camera_xyz=rr.ViewCoordinates.RDF,
                                  image_plane_distance=0.4, color=COLOR_EST), static=True)

    def set_time(self, step: int, timestamp: float):
        rr.set_time('step', sequence=step)
        rr.set_time('time', duration=timestamp)

    def log_pose(self, T_W_I_est: Isometry3D, T_W_I_gt: Optional[Isometry3D]):
        """Called at every IMU step."""
        rr.log('world/est/body', rr.Transform3D(translation=T_W_I_est.t, mat3x3=T_W_I_est.R))
        if T_W_I_gt is not None:
            rr.log('world/gt/body', rr.Transform3D(translation=T_W_I_gt.t, mat3x3=T_W_I_gt.R))

    def log_frame(self, msckf, image: np.ndarray, T_W_I_est: Isometry3D, T_W_I_gt: Optional[Isometry3D]):
        """Called after each image has been processed by the filter."""
        self.frame_count += 1
        self.est_trajectory.append(T_W_I_est.t.copy())
        if T_W_I_gt is not None: self.gt_trajectory.append(T_W_I_gt.t.copy())

        self._log_image(image)
        self._log_features(msckf)
        self._log_state(msckf, T_W_I_est)
        if self.frame_count % self.history_every == 0: self._log_history(msckf)

    def log_text(self, metrics_markdown: str, status_markdown: str):
        rr.log(METRICS, rr.TextDocument(metrics_markdown, media_type=rr.MediaType.MARKDOWN))
        rr.log(STATUS, rr.TextDocument(status_markdown, media_type=rr.MediaType.MARKDOWN))

    def _log_image(self, image: np.ndarray):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        rr.log(f'{CAMERA}/image', rr.Image(rgb).compress(jpeg_quality=self.jpeg_quality))

    def _log_features(self, msckf):
        current_camera_id = msckf.state.imu.id
        updated, gated = set(msckf.updated_feature_indices), set(msckf.gated_feature_indices)

        points, colors, radii, tracks, track_colors = [], [], [], [], []
        for index, feature in msckf.features.items():
            if len(feature.camera_indices) == 0 or feature.camera_indices[-1] != current_camera_id: continue # not seen in this image

            if index in updated: color, radius = COLOR_UPDATE, 4.0
            elif index in gated: color, radius = COLOR_GATED, 4.0
            elif len(feature.keypoints) > 1: color, radius = COLOR_TRACK, 2.5
            else: color, radius = COLOR_NEW, 2.0

            points.append(feature.keypoints[-1])
            colors.append(color)
            radii.append(radius)
            if len(feature.keypoints) > 1:
                tracks.append(np.array(feature.keypoints[-self.track_length:]))
                track_colors.append(color)

        rr.log(f'{CAMERA}/features', rr.Points2D(points, colors=colors, radii=radii))
        rr.log(f'{CAMERA}/tracks', rr.LineStrips2D(tracks, colors=track_colors, radii=0.8))

        rejected = [keypoint for keypoint, _ in msckf.rejected_keypoints]
        labels = [reason for _, reason in msckf.rejected_keypoints]
        rr.log(f'{CAMERA}/rejected', rr.Points2D(rejected, colors=COLOR_REJECTED, radii=4.0, labels=labels, show_labels=False))

    def _log_state(self, msckf, T_W_I_est: Isometry3D):
        #* Sliding window of camera clones
        window = [camera.T_W_Ci.t for camera in msckf.state.cameras.values()]
        rr.log('world/est/sliding_window', rr.Points3D(window, colors=COLOR_WINDOW, radii=0.03))
        if len(window) > 1: rr.log('world/est/sliding_window/links', rr.LineStrips3D([window], colors=COLOR_WINDOW, radii=0.008))

        #* Landmarks used in this frame's update
        points = [msckf.triangulated_points[i] for i in msckf.updated_feature_indices if i in msckf.triangulated_points]
        rr.log('world/map/current', rr.Points3D(points, colors=COLOR_UPDATE, radii=0.04))

        #* 3-sigma position uncertainty (world frame)
        sigmas = np.sqrt(np.clip(np.diag(msckf.state.covariance[12:15, 12:15]), 0, None))
        rr.log('world/est/position_3sigma', rr.Ellipsoids3D(centers=[T_W_I_est.t], half_sizes=[3*sigmas], colors=COLOR_COVARIANCE, fill_mode=3))

    def _log_history(self, msckf):
        rr.log('world/est/trajectory', rr.LineStrips3D([self.est_trajectory], colors=COLOR_EST, radii=0.01))
        if len(self.gt_trajectory) > 1: rr.log('world/gt/trajectory', rr.LineStrips3D([self.gt_trajectory], colors=COLOR_GT, radii=0.01))

        # Badly triangulated points (low parallax) can land kilometers away and would break the 3D view zoom
        landmarks = [p for p in msckf.triangulated_points.values() if np.linalg.norm(p - self.est_trajectory[-1]) < self.max_landmark_distance]
        if len(landmarks) > 0: rr.log('world/map/landmarks', rr.Points3D(landmarks, colors=COLOR_MAP, radii=0.015))
