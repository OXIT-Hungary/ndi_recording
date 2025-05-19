import multiprocessing
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from src.camera.camera import Camera
from src.config import PanoramaConfig


class PanoCamrera(Camera, multiprocessing.Process):
    """Panorama Camera Class."""

    def __init__(
        self, config: PanoramaConfig, queue, event_stop: multiprocessing.Event, save: bool = False, out_path: str = None
    ) -> None:
        Camera.__init__(self, event_stop=event_stop)
        multiprocessing.Process.__init__(self)

        self.config = config
        self.queue = queue
        self.save = save
        self.path = str(os.path.join(out_path, f"{Path(out_path).stem}_pano.mp4"))

        self.sleep_time = 1 / config.fps

    def run(self) -> None:

        x = self.config.crop[1]
        y = self.config.crop[0]
        width = self.config.crop[3] - x
        height = self.config.crop[2] - y

        ffmpeg = None
        video_cap = None
        if 'rtmp' in self.config.src or 'rtsp' in self.config.src:
            # fmt: off
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-rtsp_transport", "tcp",   # Use TCP for better stability
                    "-i", self.config.src,      # Input RTSP stream
                    "-f", "rawvideo",           # Output raw video
                    "-pix_fmt", "bgr24",        # Pixel format compatible with OpenCV
                    "-vsync", "0",              # Avoid frame duplication
                    "-an",                      # No audio
                    "-vf", f"fps={self.config.fps}, crop={width}:{height}:{x}:{y}",
                    "-fflags", "nobuffer",      # Reduce latency
                    "-probesize", "32",         # Reduce initial probe size
                    "-flags", "low_delay",      # Reduce decoding delay
                    "-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=width*height*3 * 5,
            )
            # fmt: on
        elif 'mp4' in self.config.src:
            import cv2

            video_cap = cv2.VideoCapture(self.config.src)

        ffmpeg_out = None
        if self.save:
            # fmt: off
            ffmpeg_out = subprocess.Popen(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-f", "rawvideo",
                    "-pix_fmt", "bgr24",
                    "-s", f"{width}x{height}",
                    "-r", str(self.config.fps),
                    "-hwaccel", "cuda",
                    "-hwaccel_output_format", "cuda",
                    "-i", "pipe:",
                    "-c:v", "h264_nvenc",
                    "-pix_fmt", "yuv420p",
                    "-b:v", "20000k",
                    "-preset", "fast",
                    "-profile:v", "high",
                    self.path,
                ],
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=width*height*3 * 5,
            )
            # fmt: on

        try:
            while not self.event_stop.is_set():
                start_time = time.time()

                if ffmpeg is not None:
                    raw_frame = ffmpeg.stdout.read(width * height * 3)
                    if not raw_frame:
                        continue
                        # TODO: Should we save empty image?

                    frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
                elif video_cap is not None:
                    has_frame, frame = video_cap.read()
                    if not has_frame:
                        self.event_stop.set()
                        break

                if ffmpeg_out is not None:
                    ffmpeg_out.stdin.write(frame.tobytes())
                    ffmpeg_out.stdin.flush()

                if not self.queue.full():
                    self.queue.put(frame)

                time.sleep(max(self.sleep_time - (time.time() - start_time), 0))
        except Exception as e:
            print(f"Pano Camera: {e}")
        finally:
            if ffmpeg is not None:
                ffmpeg.stdout.flush()
                ffmpeg.stdout.close()
            elif video_cap is not None:
                video_cap.release()

            if ffmpeg_out is not None:
                ffmpeg_out.stdin.flush()
                ffmpeg_out.stdin.close()
