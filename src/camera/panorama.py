import cv2
import os
from typing import List
import time
import logging

logger = logging.getLogger(__name__)


class PanoramaConfig:
    def __init__(self, pano_dict):
        self.enable = pano_dict.get("enable", False)
        self.src = pano_dict.get("src")
        self.frame_size = pano_dict.get("frame_size", [2304, 832])
        self.crop = pano_dict.get("crop", None)
        self.fps = pano_dict.get("fps", 15)


class PanoramaCamera:
    def __init__(self, config: PanoramaConfig) -> None:

        self._config = config
        self._video_capture = None

        self._video_capture = cv2.VideoCapture(self._config.src)

        if self._video_capture.isOpened():
            print(f"[INFO] Camera {self._config.src} initialized.")
        else:
            raise RuntimeError(f"[ERROR] Failed to initialize camera {self._config.src}")

    def get_frame(self):
        has_frame, frame = self._video_capture.read()
        if self._config.crop:
            frame = frame[self._config.crop[0] : self._config.crop[1], self._config.crop[2] : self._config.crop[3]]

        return frame if has_frame else None

    @property
    def frame_size(self) -> None:
        return self._config.frame_size

    @property
    def fps(self) -> None:
        return self._config.fps

    def __del__(self) -> None:
        self._video_capture.release()
