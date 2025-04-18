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
                    "-r", str(20),
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

                raw_frame = ffmpeg.stdout.read(width * height * 3)
                if not raw_frame:
                    pass
                    # raise KeyboardInterrupt()

                frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))

                if ffmpeg_out:
                    ffmpeg_out.stdin.write(frame.tobytes())
                    ffmpeg_out.stdin.flush()

                if not self.queue.full():
                    self.queue.put(frame)

                time.sleep(max(self.sleep_time - (time.time() - start_time), 0))
        except Exception as e:
            print(f"Pano Camera: {e}")
        finally:
            ffmpeg.stdout.flush()
            ffmpeg.stdout.close()

            ffmpeg_out.stdin.flush()
            ffmpeg_out.stdin.close()
