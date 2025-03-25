from src.config import PanoramaConfig
from src.camera.camera import Camera
import subprocess
import numpy as np
import multiprocessing
import time


class PanoCamrera(Camera, multiprocessing.Process):
    """Panorama Camera Class."""

    def __init__(self, config: PanoramaConfig, queue) -> None:
        Camera.__init__(self, queue=queue)
        multiprocessing.Process.__init__(self)

        self.config = config

        self.sleep_time = 1 / config.fps

    def run(self) -> None:
        self.running = True

        # fmt: off
        ffmpeg_in = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-rtsp_transport", "tcp",   # Use TCP for better stability
                "-i", self.config.src,           # Input RTSP stream
                "-f", "rawvideo",           # Output raw video
                "-pix_fmt", "bgr24",        # Pixel format compatible with OpenCV
                "-vsync", "0",              # Avoid frame duplication
                "-an",                      # No audio
                "-vf", f"fps={self.config.fps}", # Set frame rate (adjust as needed)
                "-fflags", "nobuffer",      # Reduce latency
                "-probesize", "32",         # Reduce initial probe size
                "-flags", "low_delay",      # Reduce decoding delay
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=1024**2,
        )
        # fmt: on

        try:
            while self.running:
                start_time = time.time()

                raw_frame = ffmpeg_in.stdout.read(self.config.frame_size[0] * self.config.frame_size[1] * 3)
                if not raw_frame:
                    logger.warning("No panorama frame captured.")
                    raise KeyboardInterrupt()

                frame = np.frombuffer(raw_frame, np.uint8).reshape(
                    (self.config.frame_size[1], self.config.frame_size[0], 3)
                )

                if self.config.crop:
                    frame = frame[self.config.crop[0] : self.config.crop[2], self.config.crop[1] : self.config.crop[3]]

                self.queue.put(frame)

                time.sleep(max(self.sleep_time - (time.time() - start_time), 0))
        except:
            pass
        finally:
            ffmpeg_in.stdout.flush()
            ffmpeg_in.stdout.close()

    # def get_frame(self) -> np.ndarray:
    #     """
    #     Reads a frame from RSTP stream and crops out a pre-specified location from the image.

    #     Returns: np.ndarray
    #     """

    #     raw_frame = self.ffmpeg_in.stdout.read(self.config.frame_size[0] * self.config.frame_size[1] * 3)
    #     if not raw_frame:
    #         logger.warning("No panorama frame captured.")
    #         raise KeyboardInterrupt()

    #     frame = np.frombuffer(raw_frame, np.uint8).reshape((self.config.frame_size[1], self.config.frame_size[0], 3))

    #     if self.config.crop:
    #         frame = frame[self.config.crop[0] : self.config.crop[2], self.config.crop[1] : self.config.crop[3]]

    #     return frame
