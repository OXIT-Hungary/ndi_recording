import cv2
from typing import List


class PanoramaCamera:
    def __init__(self, src, size, crop: List[int] = None, fps: int = 15) -> None:

        self._video_capture = cv2.VideoCapture(src)

        self._frame_size = size
        self._crop = crop
        self._fps = fps

    def get_frame(self):
        has_frame, frame = self._video_capture.read()
        if self._crop:
            frame = frame[self._crop[0] : self._crop[1], self._crop[2] : self._crop[3]]

        return frame if has_frame else None

    @property
    def frame_size(self) -> None:
        return self._frame_size

    @property
    def fps(self) -> None:
        return self._fps

    def __del__(self) -> None:
        self._video_capture.release()
