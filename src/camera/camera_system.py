from src.config import CameraSystemConfig
from src.camera.pano_camera import PanoCamrera
import src.camera.ptz_camera as ptz_camera

import numpy as np
import cv2
import NDIlib as ndi
import threading
import time
import onnxruntime
from collections import Counter, deque
import multiprocessing
import queue


class CameraSystem:
    """Camera System Class which incorporates and handles PTZ and Panorama cameras."""

    def __init__(self, config: CameraSystemConfig, out_path: str) -> None:
        self.config = config

        self.manager = multiprocessing.Manager()
        self.event_stop = self.manager.Event()
        self.pano_queue = self.manager.Queue(maxsize=5)

        self.cameras = {}

        if config.pano_camera.enable:
            self.cameras['pano'] = PanoCamrera(
                config=config.pano_camera,
                queue=self.pano_queue,
                event_stop=self.event_stop,
                save=True,
                out_path=out_path,
            )

        if any([cfg.enable for cfg in config.ptz_cameras.values()]):
            for name, cfg in config.ptz_cameras.items():
                if hasattr(ptz_camera, cfg.name):
                    cls = getattr(ptz_camera, cfg.name)
                    self.cameras[name] = cls(name=name, config=cfg, event_stop=self.event_stop, out_path=out_path)
                else:
                    raise ValueError(f"Class '{cfg.name}' not found in PTZCamera.py.")

        self.position = 1
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

        # if 'pano' in self.cameras:
        #     self.thread_detect_and_track = threading.Thread(target=self._detect_and_track, args=())
        #     self.thread_detect_and_track.start()
        # else:
        #     raise RuntimeError("No Panorama Camera.")

    def stop(self) -> None:
        self.event_stop.set()

        for cam in self.cameras.values():
            cam.join(timeout=5)

        # self.thread_detect_and_track.join()

    def _detect_and_track(self) -> None:
        sleep_time = 1 / 10  # 10 fps

        window = deque()
        freq_counter = Counter()

        try:
            while not self.event_stop.is_set():
                start_time = time.time()

                try:
                    frame = self.queues['pano'].get(block=False)
                except queue.Empty:
                    continue

                labels, boxes, scores = self.onnx_session.run(
                    output_names=None,
                    input_feed={
                        'images': self._transform(frame),
                        "orig_target_sizes": np.expand_dims(frame.shape[:2][::-1], axis=0),
                    },
                )

                most_populated_bucket = self._process_buckets(
                    boxes=boxes,
                    labels=labels,
                    scores=scores,
                    bucket_width=frame.shape[1] // 3,
                )
                mode = self._update_frequency(window, freq_counter, most_populated_bucket)

                if self.position != mode:
                    self.position = mode

                    for camera in [cam for name, cam in self.cameras.items() if 'ptz' in name]:
                        camera.move_to_preset(preset=ptz_camera.NUM2PRESET[self.position], speed=0x10, easing=True)

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

    def __del__(self) -> None:
        # if any([cfg.enable for cfg in self.config.ptz_cameras.values()]):
        #     ndi.find_destroy(self._ndi_find)
        pass


class CameraSystemManager:
    """Wrapper class for CameraSystem."""

    USAGE_COUNT = 0

    def __init__(self) -> None:
        pass
