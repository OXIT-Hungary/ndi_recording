import multiprocessing
import queue
import threading
import time
import os
from datetime import datetime

import cv2
import NDIlib as ndi
import numpy as np
import onnxruntime
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from enum import Enum

import src.camera.ptz_camera as ptz_camera
from src.bev import BEV
from src.camera.pano_camera import PanoCamrera
from src.config import Config
from src.utils.tmp import get_cluster_centroid
import src.utils.visualize as visualize


class Track:
    class State(Enum):
        TENTATIVE = 0
        CONFIRMED = 1
        DEAD = 2

    def __init__(self, t_id: int, x: float, y: float, dt: float) -> None:

        self.t_id = t_id
        self.kf = self._create_kalman_filter(x=x, y=y, dt=dt)

        self.confidence = 5
        self._state = Track.State.TENTATIVE

    def predict(self, has_frame: bool) -> None:
        self.kf.predict()

        if has_frame:
            self.confidence = max(self.confidence - 1, 0)

    def update(self, z, has_frame: bool) -> None:
        self.kf.update(z)

        if has_frame:
            self.confidence = min(self.confidence + 2, 15)

            if self.confidence > 8 and self._state == Track.State.TENTATIVE:
                self._state = Track.State.CONFIRMED

    def _create_kalman_filter(self, x, y, dt=1.0):
        # Initialize Kalman Filter with 4 states (x, y, vx, vy) and 2 measurements (x, y)
        kf = KalmanFilter(dim_x=4, dim_z=2)

        # State Transition Matrix (F)
        kf.F = np.array(
            [
                [1, 0, dt, 0],  # x = x + vx*dt
                [0, 1, 0, dt],  # y = y + vy*dt
                [0, 0, 1, 0],  # vx = vx
                [0, 0, 0, 1],  # vy = vy
            ]
        )

        # Measurement Function (H) - we only measure position (x, y)
        kf.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ]
        )

        # Initial State Estimate: assume starting at origin with zero velocity
        kf.x = np.array([[x], [y], [0], [0]])

        # Initial Uncertainty in the State Estimate
        kf.P = np.diag([5.0, 5.0, 25.0, 25.0])

        # Measurement Noise Covariance Matrix (R)
        kf.R = np.diag([4.0, 4.0])

        # Process Noise Covariance Matrix (Q)
        q = 1.0  # process noise magnitude
        kf.Q = (
            np.array(
                [
                    [0.25 * dt**4, 0, 0.5 * dt**3, 0],
                    [0, 0.25 * dt**4, 0, 0.5 * dt**3],
                    [0.5 * dt**3, 0, dt**2, 0],
                    [0, 0.5 * dt**3, 0, dt**2],
                ]
            )
            * q
        )

        # No control input
        kf.B = None

        return kf

    @property
    def pos(self):
        return self.kf.x[:2]

    @property
    def state(self):
        return self._state


class CameraSystem:
    """Camera System Class which incorporates and handles PTZ and Panorama cameras."""

    def __init__(self, config: Config, stream_token: str = None) -> None:
        self.config = config.camera_system
        self.out_path = f"{config.out_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.out_path, exist_ok=True)

        self.manager = multiprocessing.Manager()
        self.event_stop = self.manager.Event()
        self.pano_queue = self.manager.Queue(maxsize=1)

        self.bev = BEV(config.bev)

        self.cameras = {}
        self.camera_queues = {}
        self.camera_events = {}

        if self.config.pano_camera.enable:
            self.cameras['pano'] = PanoCamrera(
                config=self.config.pano_camera,
                queue=self.pano_queue,
                event_stop=self.event_stop,
                save=self.config.pano_camera.save,
                out_path=self.out_path,
            )

        for name, cfg in self.config.ptz_cameras.items():
            if cfg.enable:
                if hasattr(ptz_camera, cfg.name):
                    cls = getattr(ptz_camera, cfg.name)
                    self.camera_queues[name] = self.manager.Queue(maxsize=1)
                    self.camera_events[name] = self.manager.Event()
                    self.cameras[name] = cls(
                        name=name,
                        config=cfg,
                        event_stop=self.event_stop,
                        out_path=self.out_path,
                        queue_move=self.camera_queues[name],
                        event_move=self.camera_events[name],
                        stream_token=stream_token,
                    )

                else:
                    raise ValueError(f"Class '{cfg.name}' not found in PTZCamera.py.")

        self.thread_detect_and_track = None

        # logger.info("ONNX Model Device: %s", onnxruntime.get_device())
        self.onnx_session = onnxruntime.InferenceSession(
            self.config.pano_onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

    def _init_ndi(self) -> list:
        if not ndi.initialize():
            logger.error("Failed to initialize NDI.")
            return 1

        self._ndi_find = ndi.find_create_v2()
        if self._ndi_find is None:
            logger.error("Failed to create NDI find instance.")
            return 1

        sources = []
        while len(sources) < 2:
            # logger.info("Looking for sources ...")
            ndi.find_wait_for_sources(self._ndi_find, 5000)
            sources = ndi.find_get_current_sources(self._ndi_find)

        return sources

    def start(self) -> None:
        """Starts player detection on panorama frames and rotates PTZ cameras to action."""

        for camera in self.cameras.values():
            camera.start()

        if 'pano' in self.cameras:
            self.thread_detect_and_track = threading.Thread(target=self._detect_and_track, args=())
            self.thread_detect_and_track.start()
        else:
            raise RuntimeError("No Panorama Camera.")

    def stop(self) -> None:
        self.event_stop.set()

        for cam in self.cameras.values():
            cam.join(timeout=5)

        # self.thread_detect_and_track.join()

    def _detect_and_track(self) -> None:
        sleep_time = 1 / 30  # 30 fps

        kf_camera = self._create_kalman_filter(dt=sleep_time)
        tracks = []
        t_id = 1

        output = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (1500, 644))

        try:
            while not self.event_stop.is_set():
                start_time = time.time()

                try:
                    frame = self.pano_queue.get(block=False)
                except queue.Empty:
                    frame = None

                dets = []
                img_pano = np.zeros(shape=(314, 1500, 3))
                if frame is not None:
                    labels, boxes, scores = self.onnx_session.run(
                        output_names=None,
                        input_feed={
                            'images': self._transform(frame),
                            "orig_target_sizes": np.expand_dims(frame.shape[:2][::-1], axis=0),
                        },
                    )
                    img_pano = visualize.draw_boxes(frame=frame, labels=labels, boxes=boxes, scores=scores, threshold=0)

                    proj_boxes, labels, scores = self.bev.project_to_bev(boxes, labels, scores)
                    dets = proj_boxes[(labels == 2) & (scores > 0.5)].tolist()

                    img_pano = cv2.resize(img_pano, (1500, int(img_pano.shape[0] / (img_pano.shape[1] / 1500))))

                unmatched_det_inds = [i for i in range(len(dets))]

                # Propagate Tracks
                for track in tracks:
                    track.predict(has_frame=(frame is not None))

                # Associate
                track_positions = np.array([t.pos.squeeze() for t in tracks])
                track_inds, det_inds = self.associate(tracks=track_positions, dets=dets)

                # Update
                for t_ind, d_ind in zip(track_inds, det_inds):
                    tracks[t_ind].update(z=dets[d_ind], has_frame=(frame is not None))
                    unmatched_det_inds = unmatched_det_inds[:d_ind] + unmatched_det_inds[d_ind + 1 :]

                # Lifetime Management
                for ind, track in enumerate(tracks):
                    if track.confidence == 0:
                        tracks = tracks[:ind] + tracks[ind + 1 :]

                # Create new tracks
                for d_ind in unmatched_det_inds:
                    tracks.append(Track(t_id=t_id, x=dets[d_ind][0], y=dets[d_ind][1], dt=sleep_time))
                    t_id += 1

                if not len(tracks):
                    continue

                # Calculate cluster centroid for camera movement
                track_positions = np.array([t.pos.squeeze() for t in tracks if t.state == Track.State.CONFIRMED])
                cluster_center, cluster_points = get_cluster_centroid(points=track_positions, eps=5, min_samples=3)

                if cluster_center is None:
                    continue

                kf_camera.predict()
                kf_camera.update(cluster_center[0])

                img_bev = self.bev.draw(detections=track_positions, scale=15)
                img_bev = self.bev.draw_detections(img=img_bev, dets=cluster_points, scale=15, cluster=True)
                cv2.circle(
                    img_bev,
                    center=self.bev.coord_to_px(x=cluster_center[0], y=cluster_center[1], scale=15),
                    radius=3,
                    color=(0, 0, 255),
                    thickness=-1,
                )

                cv2.circle(
                    img_bev,
                    center=self.bev.coord_to_px(x=kf_camera.x[0, 0], y=0, scale=15),
                    radius=3,
                    color=(255, 0, 255),
                    thickness=-1,
                )

                new_image = np.zeros(shape=(img_bev.shape[0], img_pano.shape[1], 3), dtype=np.uint8)
                new_image[
                    :, (img_pano.shape[1] - img_bev.shape[1]) // 2 : (img_pano.shape[1] + img_bev.shape[1]) // 2, :
                ] = img_bev

                img_out = np.concatenate((img_pano, new_image), axis=0)
                output.write(img_out)

                cluster_center = kf_camera.x[0, 0]
                cluster_center = max(
                    min(cluster_center, self.bev.config.court_size[0] / 2),
                    -self.bev.config.court_size[0] / 2,
                )

                for name, ptz_cam in [(name, cam) for name, cam in self.cameras.items() if 'ptz' in name]:
                    pos_world = cluster_center if name == 'ptz1' else -cluster_center
                    pan_pos, tilt_pos = self.bev.get_pan_from_bev(pos_world, ptz_cam.presets)

                    if self.camera_queues[name].empty():
                        self.camera_queues[name].put((pan_pos, 0))
                        # self.camera_events[name].set()

                time.sleep(max(sleep_time - (time.time() - start_time), 0))

            output.release()
        except KeyboardInterrupt:
            logger.info("Keyboard Interrupt received.")

    def is_running(self):
        return not self.event_stop.is_set()

    def associate(self, tracks, dets, VI=None):
        if len(tracks) == 0 or len(dets) == 0:
            return [], []

        if VI is None:
            cost_matrix = cdist(tracks, dets, metric='euclidean')
        else:
            cost_matrix = cdist(tracks, dets, metric='mahalanobis', VI=VI)

        return linear_sum_assignment(cost_matrix)

    def _create_kalman_filter(self, dt=1.0):
        # Initialize the Kalman Filter
        kf = KalmanFilter(dim_x=2, dim_z=1)

        # State Transition Matrix (F)
        kf.F = np.array([[1, dt], [0, 1]])

        # Measurement Function (H) - we only measure position
        kf.H = np.array([[1, 0]])

        # Initial State Estimate
        kf.x = np.array([[0], [0]])  # initial position and velocity

        # Initial Uncertainty
        kf.P *= 50.0  # high uncertainty in initial state

        # Measurement Noise Covariance (R)
        kf.R = np.array([[100]])  # tune this: smaller = more trust in measurements

        # Process Noise Covariance (Q)
        kf.Q = np.eye(2) * 0.1

        # Initial Estimate Covariance
        kf.B = 0  # no control input

        return kf

    def _transform(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        return np.expand_dims(np.transpose(frame, (2, 0, 1)), axis=0)

    def __del__(self) -> None:
        pass


class CameraSystemManager:
    """Wrapper class for CameraSystem."""

    USAGE_COUNT = 0

    def __init__(self) -> None:
        pass