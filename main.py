import logging
import os
import subprocess
import time

from datetime import datetime, timedelta
from typing import Tuple
from tqdm import tqdm

from src.camera.camera_system import CameraSystem


out_path = f"{os.getcwd()}/output/{datetime.now().strftime('%Y%m%d_%H%M')}"
os.makedirs(out_path, exist_ok=True)


def create_logger():
    l = logging.getLogger("ndi_logger")
    l.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f"{out_path}/run.log", mode="w")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="{asctime} - [{levelname}]: {message}",
        style="{",
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    l.addHandler(console_handler)
    l.addHandler(file_handler)

    return l


logger = create_logger()


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


def parse_arguments(args) -> Tuple[datetime]:

    now = datetime.now()

    start_time = now
    if args.start_time:
        splt = args.start_time.split("_")
        if len(splt) == 1:
            h, m = splt[0].split(":")
            start_time = datetime.strptime(f"{now.year}.{now.month}.{now.day}_{h}:{m}", "%Y.%m.%d_%H:%M")
        else:
            start_time = datetime.strptime(args.start_time, "%Y.%m.%d_%H:%M")

    end_time = None
    if args.end_time:
        splt = args.end_time.split("_")
        if len(splt) == 1:
            h, m = splt[0].split(":")
            end_time = datetime.strptime(f"{now.year}.{now.month}.{now.day}_{h}:{m}", "%Y.%m.%d_%H:%M")
        else:
            end_time = datetime.strptime(args.start_time, "%Y.%m.%d_%H:%M")
    elif args.duration:
        duration = datetime.strptime(args.duration, "%H:%M").time()
        end_time = start_time + timedelta(hours=duration.hour, minutes=duration.minute)
    else:
        duration = datetime.strptime("01:45", "%H:%M").time()
        end_time = start_time + timedelta(hours=duration.hour, minutes=duration.minute)

    return start_time, end_time


def main(start_time: datetime, end_time: datetime) -> int:

    schedule(start_time, end_time)

    camera_system = CameraSystem()
    camera_system.start_tracking()

    writers = [
        VideoWriter(path="ptz1.mp4", resolution="1920x1080", bitrate=40000),  # PTZ 1
        VideoWriter(path="ptz2.mp4", resolution="1920x1080", bitrate=40000),  # PTZ 2
        VideoWriter(path="pano.mp4", resolution="", bitrate=6000),  # Pano
    ]

    try:
        delta_time = int((end_time - start_time).total_seconds())
        with tqdm(total=delta_time, bar_format="{l_bar}{bar} [Elapsed: {elapsed}, Remaining: {remaining}]") as progress:
            for _ in range(delta_time):
                frames = camera_system.get_frames()
                for idx, writer in enumerate(writers):
                    writer.write(frames[idx])

                time.sleep(1)
                progress.update(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Terminating processes...")

    finally:
        camera_system.stop_tracking()

    logger.info("Finished recording.")

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="NDI Stream Recorder", description="Schedule a script to run based on time.")

    parser.add_argument("--start_time", type=str, help="Start time in HH:MM format. e.g. (18:00)", required=False)
    parser.add_argument("--end_time", type=str, help="End time in HH:MM format. e.g. (18:00)", required=False)
    parser.add_argument("--duration", type=str, help="Duration in HH:MM format. e.g. (18:00)", required=False)

    args = parse_arguments(parser.parse_args())

    main(*args)
