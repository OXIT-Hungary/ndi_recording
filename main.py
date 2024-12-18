import logging
import os
import subprocess
import time
from collections import Counter, deque
from typing import List

import cv2
import NDIlib as ndi
import numpy as np
import onnxruntime
from multiprocess import Event


def process_buckets(boxes, labels, scores, bucket_width):
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
    bucket_width = 2200 // 3

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

            most_populated_bucket = process_buckets(boxes, labels, scores, bucket_width)
            mode = update_frequency(window, freq_counter, most_populated_bucket)

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
                os.path.join(self.path, f"cam{self.idx}.mp4"),
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
    receiver = NDIReceiver(src, idx, path, codec, fps)

    logger.info(f"NDI Receiver {idx} created.")

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
