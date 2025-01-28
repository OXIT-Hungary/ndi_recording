import NDIlib as ndi
import onnxruntime

from enum import Enum
from multiprocess import Event, Process

from ptz import PTZCamera
from panorama import PanoramaCamera


class CameraSystem:
    class Position(Enum):
        LEFT = 0
        CENTER = 1
        RIGHT = 2

    def __init__(self, onnx_file: str) -> None:
        # Setup panorama camera
        self.panorama_camera = PanoramaCamera(src="")

        # Setup PTZ cameras
        self.ptz_cameras = self._init_ptz_cameras()
        for ptz_cam in self.ptz_cameras:
            ptz_cam.move()  # TODO: Set initial coordinates

        self.position = CameraSystem.Position.CENTER

        print(f"ONNX Model Device: {onnxruntime.get_device()}")
        self.onnx_session = onnxruntime.InferenceSession(
            onnx_file, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        self.event_stop = Event()

        self.panorama_process = Process(target=self._pano_process, args=(start_event))

        self.ptz_processes = []
        for ptz_camera in self.ptz_cameras:
            p = Process(target=self._ndi_process, args=(ptz_camera))
            self.ptz_processes.append(p)
            p.start()

        self.pano_frame = None
        self.ptz_frames = [None] * len(self.ptz_cameras)

    def start_tracking(self) -> None:
        self.panorama_process.start()

    def stop_tracking(self) -> None:
        self.event_stop.set()

        self.panorama_process.join()
        self.panorama_process.kill()

    def get_frames(self):
        return self.pano_frame, self.ptz_frames

    def _init_ptz_cameras(self):
        if not ndi.initialize():
            logger.error("Failed to initialize NDI.")
            return 1

        self.ndi_find = ndi.find_create_v2()
        if self.ndi_find is None:
            logger.error("Failed to create NDI find instance.")
            return 1

        logger.info("Looking for sources ...")
        ndi.find_wait_for_sources(self.ndi_find, 8000)
        sources = ndi.find_get_current_sources(self.ndi_find)

        cams = []
        for idx, source in enumerate(sources):
            cams.append(PTZCamera(src=source, idx=idx))

        return cams

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

    def _pano_process(
        self,
        start_event: Event,
    ):
        """ """
        import cv2
        import numpy as np
        import time
        from collections import deque, Counter

        def _transform(frame: np.ndarray) -> np.ndarray:
            frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            return np.expand_dims(np.transpose(frame, (2, 0, 1)), axis=0)

        sleep_time = 1 / self.panorama_camera.fps
        bucket_width = 2304 // 3

        window = deque()
        freq_counter = Counter()

        start_event.set()

        while not self.event_stop.is_set():
            frame = self.panorama_camera.get_frame()
            if frame is not None:
                self.pano_frame = frame.copy()

                labels, boxes, scores = self.onnx_session.run(
                    output_names=None,
                    input_feed={
                        'images': _transform(frame),
                        "orig_target_sizes": self.panorama_camera.frame_size,
                    },
                )

                most_populated_bucket = self._process_buckets(boxes, labels, scores, bucket_width)
                mode = self._update_frequency(window, freq_counter, most_populated_bucket)

                if self.position != mode:
                    # TODO: Implement camera movement
                    pass
            else:
                print("No panorama frame captured.")

            time.sleep(sleep_time)

    def _ndi_process(self, camera: PTZCamera) -> None:
        while not self.event_stop.is_set():
            self.ptz_frames[camera.idx] = camera.get_frame()

    def __del__(self) -> None:
        for process in self.ptz_processes:
            if process.is_alive():
                process.join()

        ndi.find_destroy(self.ndi_find)
