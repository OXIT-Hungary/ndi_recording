import logging
import os
import subprocess
import time
from collections import Counter, deque
from typing import List
from pathlib import Path

import cv2
import NDIlib as ndi
import numpy as np
import onnxruntime
from multiprocess import Event



def split_the_waterpolo_court_panorama_image(n, first_percent, middle_percent):
    # n: width of the panorama image
    # first_percent: integer between 20-45%
    # first test: 40%-20%-40%; try to reduce the middle band
    
    first_part = int(n * (first_percent / 100))
    middle_part = int(n * (middle_percent / 100))
    third_part = n - first_part - middle_part
    
    return first_part, middle_part, third_part


def decide_band_with_most_players(boxes, labels, scores, first_band_end, middle_band_end):
    bboxes_player = boxes[(labels == 2) & (scores > 0.5)]

    if len(bboxes_player) == 0:
        return 1, [0, 0, 0]  # No players, default to middle band

    centers_x = (bboxes_player[:, 0] + bboxes_player[:, 2]) / 2
    
    band_counts = [0, 0, 0]  # Index 0: first_band, Index 1: middle_band, Index 2: third_band
    
    for x in centers_x:
        if x < first_band_end:
            band_counts[0] += 1  
        elif x < middle_band_end:
            band_counts[1] += 1  
        else:
            band_counts[2] += 1  
    
    
    most_players_band_index = np.argmax(band_counts)
    
    return most_players_band_index, band_counts


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


def pano_process(
    url: str,
    ptz_urls: List,
    onnx_file: str,
    stop_event: Event,
    start_event: Event,
    logger: logging.Logger,
    fps: int = 15,
):
    """ """

    position = 1

    frame_size = np.array([[2200, 730]])
    sleep_time = 1 / fps
    
    first_percent, middle_percent = 41, 18

    first_band_end, middle_part, _ = split_the_waterpolo_court_panorama_image(frame_size[0,0], first_percent, middle_percent)

    middle_band_end = first_band_end + middle_part

    onnx_session = onnxruntime.InferenceSession(onnx_file, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    logger.info(f"ONNX Model Device: {onnxruntime.get_device()}")

    window = deque()
    freq_counter = Counter()

    video_capture = cv2.VideoCapture(url)

    start_event.set()
    logger.info(f"Process Pano - Event Set!")
    try:
        while not stop_event.is_set():
            ret, frame = video_capture.read()

            if not ret:
                logger.warning(f"No panorama frame captured.")
                raise KeyboardInterrupt()

            frame = frame[420:1150, 1190:3390]
            img = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

            labels, boxes, scores = onnx_session.run(
                output_names=None,
                input_feed={
                    'images': img,
                    "orig_target_sizes": frame_size,
                },
            )

            # most_populated_bucket = process_buckets(boxes, labels, scores, bucket_width)
            most_players_band, _ = decide_band_with_most_players(boxes, labels, scores, first_band_end, middle_band_end)

            mode = update_frequency(window, freq_counter, most_players_band)

            # draw(Image.fromarray(frame), labels, boxes, scores, mode, bucket_width)

            if position != mode:
                position = mode
                for url in ptz_urls:
                    command = (
                        rf'szCmd={{'
                        rf'"SysCtrl":{{'
                        rf'"PtzCtrl":{{'
                        rf'"nChanel":0,"szPtzCmd":"preset_call","byValue":{mode}'
                        rf'}}'
                        rf'}}'
                        rf'}}'
                    )

                    subprocess.run(
                        [
                            "curl",
                            f"http://{url}/ajaxcom",
                            "--data-raw",
                            command,
                        ],
                        check=False,
                        capture_output=False,
                        text=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

            time.sleep(sleep_time)

    except KeyboardInterrupt:
        video_capture.release()

    logger.info(f"RTSP Receiver Process stopped.")


class NDIReceiver:
    def __init__(self, src, idx: int, path, logger: logging.Logger, codec="h264_nvenc", fps: int = 30) -> None:
        self.idx = idx
        self.codec = codec
        self.fps = fps
        self.path = path
        self.logger = logger

        self.receiver = self.create_receiver(src)
        self.ffmpeg_process = self.start_ffmpeg_process()

    def create_receiver(self, src):

        ndi_recv_create = ndi.RecvCreateV3()
        ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
        receiver = ndi.recv_create_v3(ndi_recv_create)
        if receiver is None:
            raise RuntimeError("Failed to create NDI receiver")
        ndi.recv_connect(receiver, src)

        return receiver

    def get_frame(self):

        t, v, _, _ = ndi.recv_capture_v3(self.receiver, 1000)
        frame = None
        if t == ndi.FRAME_TYPE_VIDEO:
            # logger.info("Frame received")
            frame = np.copy(v.data[:, :, :3])
            ndi.recv_free_video_v2(self.receiver, v)
            # cv2.imwrite('output/asd.png', frame)
            # print(frame.shape)

        return frame, t

    def start_ffmpeg_process(self):
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
                "1920x1080",
                "-r",
                str(self.fps),
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-i",
                "pipe:",
                "-c:v",
                self.codec,
                "-pix_fmt",
                "yuv420p",
                "-b:v",
                "40000k",
                "-preset",
                "fast",
                "-profile:v",
                "high",
                self.path,
            ],
            stdin=subprocess.PIPE,
        )

    def stop(self) -> None:
        if self.ffmpeg_process.stdin:
            try:
                self.ffmpeg_process.stdin.flush()
                self.ffmpeg_process.stdin.close()
            except BrokenPipeError as e:
                self.logger.error(f"Broken pipe error while closing stdin: {e}")

        self.ffmpeg_process.wait()


def ndi_receiver_process(
    src, idx: int, path, logger: logging.Logger, stop_event: Event, codec: str = "h264_nvenc", fps: int = 30
):
    path = os.path.join(path, f"{Path(path).stem}_cam{idx}.mp4")
    receiver = NDIReceiver(src, idx, path, logger, codec, fps)

    logger.info(f"NDI Receiver {idx} created. Saving data to {path}")

    try:
        while not stop_event.is_set():
            frame, t = receiver.get_frame()
            if frame is not None:
                try:
                    receiver.ffmpeg_process.stdin.write(frame.tobytes())
                    receiver.ffmpeg_process.stdin.flush()
                except BrokenPipeError as e:
                    logger.error(f"Broken pipe error while writing frame: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error in NDI Receiver Process {idx}: {e}")
                    break
            else:
                logger.warning(f"No video frame captured. Frame type: {t}")
    except KeyboardInterrupt:
        receiver.stop()

    logger.info(f"NDI Receiver Process {receiver.idx} stopped.")


if __name__ == "__main__":
    from multiprocess import Event, Process
    from datetime import datetime

    from app.core.utils.dir_creator import get_recording_dir_from_datetime
    from app.core.utils.logger import get_recording_logger

    start_time = datetime.now()

    recording_dir = get_recording_dir_from_datetime(start_time)
    logger = get_recording_logger(start_time)

    stop_event = Event()

    ndi_receiver_process("", 0, recording_dir, logger, stop_event)
