import multiprocessing
import queue
import threading
import time

import cv2
import NDIlib as ndi
import numpy as np
import onnxruntime

import src.camera.ptz_camera as ptz_camera
from src.bev import BEV
from src.camera.pano_camera import PanoCamrera
from src.config import Config
from src.player_tracker import Tracker
from src.utils.tmp import get_cluster_centroid
from src.utils.visualize import debug_visualization


class CameraSystem:
    """Camera System Class which incorporates and handles PTZ and Panorama cameras."""

    def __init__(self, config: Config, out_path: str, stream_token=None) -> None:
        self.config = config.camera_system
        self.out_path = out_path
        self.new_centroid = np.array([0, 0])

        self.manager = multiprocessing.Manager()
        self.event_stop = self.manager.Event()
        self.pano_queue = self.manager.Queue(maxsize=5)

        self.bev = BEV(config.bev)

        # self.tracker = Tracker(max_age=25, min_hits=3)  # TODO: Refactor and implement bev tracking

        self.centroid = np.array([0, 0])

        self.cameras = {}
        self.camera_queues = {}
        self.camera_events = {}

        if config.pano_camera.enable:
            self.cameras['pano'] = PanoCamrera(
                config=config.pano_camera,
                queue=self.pano_queue,
                event_stop=self.event_stop,
                save=True,
                out_path=out_path,
            )

        for name, cfg in config.ptz_cameras.items():
            if cfg.enable:
                if hasattr(ptz_camera, cfg.name):
                    cls = getattr(ptz_camera, cfg.name)
                    self.cameras[name] = cls(
                        name=name, config=cfg, event_stop=self.event_stop, out_path=out_path, stream_token=stream_token
                    )
                    self.camera_queues[name] = self.manager.Queue(maxsize=1)
                    self.camera_events[name] = self.manager.Event()
                else:
                    raise ValueError(f"Class '{cfg.name}' not found in PTZCamera.py.")

        self.thread_detect_and_track = None

        # logger.info("ONNX Model Device: %s", onnxruntime.get_device())
        self.onnx_session = onnxruntime.InferenceSession(
            config.pano_onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
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
        sleep_time = 1 / 10  # 10 fps

        try:
            while not self.event_stop.is_set():
                start_time = time.time()

                try:
                    frame = self.pano_queue.get(block=False)
                except queue.Empty:
                    continue

                labels, boxes, scores = self.onnx_session.run(
                    output_names=None,
                    input_feed={
                        'images': self._transform(frame),
                        "orig_target_sizes": np.expand_dims(frame.shape[:2][::-1], axis=0),
                    },
                )

                # from src.utils.visualize import draw
                # draw(image=frame, labels=labels,boxes=boxes, scores=scores)

                proj_boxes, labels, scores = self.bev.project_to_bev(boxes, labels, scores)
                proj_players = proj_boxes[(labels == 2) & (scores > 0.5)]

                gravity_center = get_cluster_centroid(proj_players)
                if gravity_center is not None:
                    gravity_center[0] = max(
                        min(gravity_center[0], self.config.bev.court_size[0] / 2), -self.config.bev.court_size[0] / 2
                    )

                    if abs(gravity_center[0] - self.centroid[0]) > self.config.track_threshold:
                        self.centroid = gravity_center

                        for name, ptz_cam in [(name, cam) for name, cam in self.cameras.items() if 'ptz' in name]:
                            pos = gravity_center[0] if name == 'ptz1' else -gravity_center[0]
                            pan_hex, tilt_hex = self.bev.get_pan_from_bev(pos, ptz_cam.presets)

                            if not self.camera_queues[name].full():
                                self.camera_queues[name].put((pan_hex, tilt_hex))
                                self.camera_events[name].set()

                    if self.debug_mode and self.centroid is not None and gravity_center is not None:
                        debug_visualization(self.debug_idx, self.centroid, proj_players, gravity_center)
                        self.debug_idx = self.debug_idx + 1

                time.sleep(max(sleep_time - (time.time() - start_time), 0))

        except KeyboardInterrupt:
            logger.info("Keyboard Interrupt received.")

    def _process_buckets(self, boxes, labels, scores, bucket_width):
        """ """

        buckets = {0: 0, 1: 0, 2: 0}
        bboxes_player = boxes[(labels == 2) & (scores > 0.5)]
        centers_x = (bboxes_player[:, 0] + bboxes_player[:, 2]) / 2

        for center_x in centers_x:
            bucket_idx = center_x // bucket_width
            buckets[bucket_idx] += 1

        return max(buckets, key=lambda k: buckets[k])

    def _update_frequency(self, window, freq_counter, bucket, max_window_size=10):
        """
        Update the sliding window and frequency counter to track the most frequent bucket.
        """

        window.append(bucket)
        freq_counter[bucket] += 1

        if len(window) > max_window_size:
            oldest = window.popleft()
            freq_counter[oldest] -= 1
            if freq_counter[oldest] == 0:
                del freq_counter[oldest]

        # Return the most common bucket
        return freq_counter.most_common(1)[0][0]

    def _transform(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        return np.expand_dims(np.transpose(frame, (2, 0, 1)), axis=0)

    def _lerp(self, t, times, points):
        dx = points[1][0] - points[0][0]
        dy = points[1][1] - points[0][1]
        dt = (t - times[0]) / (times[1] - times[0])
        return np.array([dt * dx + points[0][0], dt * dy + points[0][1]])

    def _move_centroid_smoothly(self, current_pos, new_pos):
        return self._lerp(10, [1, 100], [current_pos, new_pos])

    def __del__(self) -> None:
        pass


class CameraSystemManager:
    """Wrapper class for CameraSystem."""

    USAGE_COUNT = 0

    def __init__(self) -> None:
        pass
