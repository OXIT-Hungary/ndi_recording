import multiprocessing
import os
import subprocess
import time
from pathlib import Path
import queue

import cv2
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

        crop = self.config.crop
        if crop:
            x, y = crop[1], crop[0]
            width, height = crop[3] - crop[1], crop[2] - crop[0]
        else:
            width = self.config.frame_size[0]
            height = self.config.frame_size[1]

        ffmpeg = video_cap = None
        if 'rtmp' in self.config.src or 'rtsp' in self.config.src:
            vf_filter = f"fps={self.config.fps},scale={width}:{height}"

            # fmt: off
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-rtsp_transport", "tcp",   # Use TCP for better stability
                    "-i", self.config.src,
                    "-f", "rawvideo",           # Output raw video
                    "-pix_fmt", "bgr24",
                    "-vsync", "0",              # Avoid frame duplication
                    "-an",                      # No audio
                    "-vf", vf_filter,
                    "-fflags", "nobuffer",      # Reduce latency
                    "-probesize", "32",         # Reduce initial probe size
                    "-flags", "low_delay",      # Reduce decoding delay
                    "-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.config.frame_size[0] * self.config.frame_size[1] * 3 * 5,
            )
            # fmt: on
        elif 'mp4' in self.config.src:
            video_cap = cv2.VideoCapture(self.config.src)

        ffmpeg_out = None  # TODO: Check out shape if crop
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
                bufsize=self.config.frame_size[0]*self.config.frame_size[1]*3 * 5,
            )
            # fmt: on

        try:
            while not self.event_stop.is_set():
                start_time = time.time()

                if ffmpeg is not None:
                    raw_frame = ffmpeg.stdout.read(self.config.frame_size[0] * self.config.frame_size[1] * 3)

                    if raw_frame:
                        frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
                        frame = self.undist_image(frame)
                    else:
                        frame = np.zeros(shape=(height, width, 3), dtype=np.uint8)
                elif video_cap is not None:
                    has_frame, frame = video_cap.read()

                    if not has_frame:
                        self.event_stop.set()
                        break

                # Crop Frame
                if self.config.crop:
                    frame = frame[y : y + height, x : x + width, :]

                # Save Frame
                if ffmpeg_out:
                    ffmpeg_out.stdin.write(frame.tobytes())
                    ffmpeg_out.stdin.flush()

                if self.queue.full():
                    self.queue.get()
                self.queue.put_nowait(frame)

                time.sleep(max(self.sleep_time - (time.time() - start_time), 0))
        except Exception as e:
            print(f"Pano Camera: {e}")
        finally:
            if ffmpeg:
                ffmpeg.stdout.flush()
                ffmpeg.stdout.close()
            elif video_cap:
                video_cap.release()

            if ffmpeg_out:
                ffmpeg_out.stdin.flush()
                ffmpeg_out.stdin.close()

    def undist_image(self, frame):

        undistorted = cv2.remap(
            frame, self.config.map_x, self.config.map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
        )

        return undistorted
