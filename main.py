import subprocess
import time

from datetime import datetime
from tqdm import tqdm
import onnxruntime
import cv2
import numpy as np
from collections import deque, Counter

from src.config import Config, load_config
from src.camera.camera_system import CameraSystem
from src.utils.logger import setup_logger


logger = None


class VideoWriter:
    def __init__(
        self,
        path: str,
        resolution: str = "1920x1080",
        codec: str = "h264_nvenc",
        fps: int = 30,
        bitrate: int = 30000,
        *args,
    ) -> None:

        self._ffmpeg_process = self._start_ffmpeg_process(
            path=path,
            resolution=resolution,
            codec=codec,
            fps=fps,
            bitrate=bitrate,
        )

    def _start_ffmpeg_process(
        self,
        path: str,
        resolution: str = "1920x1080",
        codec: str = "h264_nvenc",
        fps: int = 30,
        bitrate: int = 30000,
    ):
        return subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                resolution,
                "-r",
                str(fps),
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-i",
                "pipe:",
                "-c:v",
                codec,
                "-pix_fmt",
                "yuv420p",
                "-b:v",
                f"{bitrate}k",
                "-preset",
                "fast",
                "-profile:v",
                "high",
                path,
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame) -> None:
        try:
            self._ffmpeg_process.stdin.write(frame.tobytes())
            self._ffmpeg_process.stdin.flush()
        except BrokenPipeError as e:
            logger.error(f"Broken pipe error while writing frame: {e}")

    def __del__(self) -> None:
        if self._ffmpeg_process.stdin:
            try:
                self._ffmpeg_process.stdin.flush()
                self._ffmpeg_process.stdin.close()
            except BrokenPipeError as e:
                self.logger.error(f"Broken pipe error while closing stdin: {e}")

        self._ffmpeg_process.wait()


def schedule(start_time: datetime, end_time: datetime) -> None:

    sleep_time = int((start_time - datetime.now()).total_seconds())
    if sleep_time <= 0:
        return

    hours, remainder = divmod(sleep_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info(
        f"Waiting for {hours:02d}:{minutes:02d}:{seconds:02d} (hh:mm:ss) until {end_time.strftime('%Y.%m.%d %H:%M:%S')}."
    )

    with tqdm(total=sleep_time, bar_format="{l_bar}{bar} [Elapsed: {elapsed}, Remaining: {remaining}]") as progress:
        for _ in range(int(sleep_time)):
            time.sleep(1)
            progress.update(1)

    logger.info(f"Finished waiting.")


def transform(frame: np.ndarray) -> np.ndarray:
    frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return np.expand_dims(np.transpose(frame, (2, 0, 1)), axis=0)


def process_buckets(boxes, labels, scores, bucket_width):
    """ """
    buckets = {0: 0, 1: 0, 2: 0}
    bboxes_player = boxes[(labels == 2) & (scores > 0.5)]
    centers_x = (bboxes_player[:, 0] + bboxes_player[:, 2]) / 2

    for center_x in centers_x:
        bucket_idx = center_x // bucket_width
        buckets[bucket_idx] += 1

    return max(buckets, key=lambda k: buckets[k])


def update_frequency(window, freq_counter, bucket, max_window_size=10):
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


def main(config: Config) -> int:

    schedule(config.schedule.start_time, config.schedule.end_time)

    camera_system = CameraSystem(config=config.camera_system)
    camera_system.start()

    writers = {
        'ptz1': VideoWriter(path="ptz1.mp4", resolution="1920x1080", bitrate=40000),  # PTZ 1
        'ptz2': VideoWriter(path="ptz2.mp4", resolution="1920x1080", bitrate=40000),  # PTZ 2
        'pano': VideoWriter(path="pano.mp4", resolution="", bitrate=6000),  # Pano
    }

    try:
        logger.debug(f"ONNX Model Device: {onnxruntime.get_device()}")
        onnx_session = onnxruntime.InferenceSession(
            config.pano_onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        window = deque()
        freq_counter = Counter()

        delta_time = int((config.schedule.end_time - config.schedule.start_time).total_seconds())
        with tqdm(total=delta_time, bar_format="{l_bar}{bar} [Elapsed: {elapsed}, Remaining: {remaining}]") as progress:
            for _ in range(delta_time):
                frames = camera_system.get_frames()
                for writer_name, writer in writers.items():
                    writer.write(frames[writer_name])

                # Pano detection
                labels, boxes, scores = onnx_session.run(
                    output_names=None,
                    input_feed={
                        'images': transform(frames['pano']),
                        "orig_target_sizes": config.camera_system.pano_camera.frame_size,
                    },
                )

                most_populated_bucket = process_buckets(
                    boxes=boxes,
                    labels=labels,
                    scores=scores,
                    bucket_width=config.camera_system.pano_camera.frame_size[0] // 3,
                )
                mode = update_frequency(window, freq_counter, most_populated_bucket)
                camera_system.move_to_preset(pos=CameraSystem.Position(mode))

                time.sleep(1)
                progress.update(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Terminating processes...")

    except Exception as e:
        logger.error(e)
        print(e)

    finally:
        camera_system.stop()

    logger.info("Finished recording.")

    return 0


if __name__ == "__main__":
    import argparse
    import multiprocessing

    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(prog="NDI Stream Recorder", description="Schedule a script to run based on time.")
    parser.add_argument("--config", type=str, help="Config path.", required=False, default="./config.yaml")

    args = parser.parse_args()
    cfg = load_config(args.config)

    logger = setup_logger(log_dir=cfg.out_path)

    main(config=cfg)
