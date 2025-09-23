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


import src.camera.ptz_camera as ptz_camera
from src.game import Game
from src.camera.pano_camera import PanoCamrera
from src.config import Config
from src.utils.tmp import get_cluster_centroid
import src.utils.visualize as visualize


class CameraSystem:
    """Camera System Class which incorporates and handles PTZ and Panorama cameras."""

    def __init__(self, config: Config, stream_token: str = None) -> None:
        self.config = config.camera_system
        self.out_path = f"{config.out_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.out_path, exist_ok=True)

        self.manager = multiprocessing.Manager()
        self.event_stop = self.manager.Event()
        self.pano_queue = self.manager.Queue(maxsize=1)

        self.game = Game(config.bev)

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

        self.thread_detect_and_track.join()

    def _detect_and_track(self) -> None:
        sleep_time = 1 / 5

        # output = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (1500, 593))

        try:
            cam_pos = 0
            while not self.event_stop.is_set():
                start_time = time.time()

                try:
                    frame_pano = self.pano_queue.get(block=False)
                except queue.Empty:
                    frame_pano = None

                labels, boxes, scores = [], [], []
                if frame_pano:
                    labels, boxes, scores = self.onnx_session.run(
                        output_names=None,
                        input_feed={
                            'images': self._transform(frame_pano),
                            "orig_target_sizes": np.expand_dims(frame_pano.shape[:2][::-1], axis=0),
                        },
                    )

                self.game.update(labels, boxes, scores)

                # Calculate cluster centroid for camera movement
                tracks = self.game.tracks
                track_positions = np.array([t.pos.squeeze() for t in tracks])

                cluster_center, cluster_points, mask = get_cluster_centroid(
                    points=track_positions, eps=5, min_samples=3
                )

                direction = 0
                for track, m in zip(tracks, mask):
                    if m:
                        if track.get_direction()[0] > 0 and track.speed[0] > 0.3:
                            direction += 1
                        else:
                            direction -= 1

                cluster_center = max(
                    min(cluster_center[0], self.game.court_width / 2),
                    -self.game.court_width / 2,
                )

                for border in [-self.game.court_width / 3 / 2, self.game.court_width / 3 / 2]:
                    if abs(cluster_center - border) < 3:
                        if border > 0 and direction > 2:
                            cam_pos = 9
                        elif border < 0 and direction < -2:
                            cam_pos = -9
                        elif border > 0 and direction < -2:
                            cam_pos = 0
                        elif border < 0 and direction > 2:
                            cam_pos = 0

                for name, ptz_cam in [(name, cam) for name, cam in self.cameras.items() if 'ptz' in name]:
                    pos_world = cam_pos if name == 'ptz1' else -cam_pos
                    pan_pos, tilt_pos = self.game.get_pan_from_bev(pos_world, ptz_cam.presets)

                    if not self.camera_queues[name].empty():
                        self.camera_queues[name].get()
                        # self.camera_events[name].set()

                    self.camera_queues[name].put((pan_pos, 0))

                time.sleep(max(sleep_time - (time.time() - start_time), 0))

            # output.release()
        except KeyboardInterrupt:
            print.info("Keyboard Interrupt received.")

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
