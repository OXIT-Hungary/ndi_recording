import math
import os
import subprocess
import time
from collections import Counter, deque
from datetime import datetime
from multiprocessing import Event, Process
from pathlib import Path

import cv2
import NDIlib as ndi
import numpy as np
import onnxruntime
from PIL import Image, ImageDraw
from tqdm import tqdm

import src.visca as visca
from src.config import Config, PanoramaConfig, load_config
from src.utils.logger import setup_logger

from bev_main import BEV

PAN_TIMES = [
    136.56848526000977,
    69.01856327056885,
    54.386627197265625,
    45.766037940979004,
    37.671316385269165,
    29.31937313079834,
    24.364627599716187,
    22.020536422729492,
    20.19959259033203,
    16.56476140022278,
    13.843563079833984,
    12.680556058883667,
    11.287461519241333,
    10.541606664657593,
    10.029051780700684,
    9.134373188018799,
    8.485095500946045,
    7.30612587928772,
    6.574438095092773,
    5.858522176742554,
    4.796533584594727,
    4.0511314868927,
    3.6801223754882812,
    3.3116557598114014,
]

TILT_TIMES = [
    75.15143918991089,
    37.944077014923096,
    32.19101023674011,
    24.17175054550171,
    19.466105937957764,
    16.21293330192566,
    13.038465738296509,
    10.229816198348999,
    9.39140510559082,
    8.00383973121643,
    7.039175033569336,
    6.424523830413818,
    5.612829208374023,
    5.345156669616699,
    4.974465847015381,
    4.504214286804199,
    4.234630107879639,
    3.8986916542053223,
    3.5139451026916504,
    3.215477466583252,
]

PAN_SPEEDS = {key + 1: 340 / value for key, value in enumerate(PAN_TIMES)}
TILT_SPEEDS = {key + 1: 120 / value for key, value in enumerate(TILT_TIMES)}

class2color = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0)}
class2str = {1: 'Goalkeeper', 2: 'Player', 3: 'Referee'}

num2preset = {0: 'left', 1: 'center', 2: 'right'}


def draw(image, labels, boxes, scores, bucket_id, bucket_width, thrh=0.5):
    draw = ImageDraw.Draw(image)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    scr = scores
    lab = labels[scr > thrh]
    box = boxes[scr > thrh]

    left = bucket_id * bucket_width
    right = (bucket_id + 1) * bucket_width
    draw_overlay.rectangle([left, 0, right, image.height], fill=(0, 128, 255, 50))

    for box, label, score in zip(boxes, labels, scores):
        if score > thrh:
            draw.rectangle(box.tolist(), outline=class2color[label], width=2)
            draw.text((box[0], box[1]), text=class2str[label], fill="blue")

    blended = Image.alpha_composite(image.convert("RGBA"), overlay)
    cv2.imshow("Image", np.array(blended))
    cv2.waitKey(1)


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


def find_closest_key(value_dict, target_value):
    return min(value_dict, key=lambda k: abs(value_dict[k] - target_value))


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


def calculate_intermediate_positions(start, end, steps):
    """
    Generates evenly distributed intermediate positions with cubic easing.
    Ensures smooth acceleration and deceleration.
    """

    if abs(end - start) > 3000:
        # If the difference is too large, adjust the direction
        if end > start:
            start += 65536  # Move start to the next "cycle"
        else:
            end += 65536  # Move end to the next "cycle"

    # Generate normalized steps (0 to 1)
    t_values = np.linspace(0, 1, steps)
    t_values = 3 * t_values**2 - 2 * t_values**3  # Cubic easing

    # Compute positions
    positions = [int(start + (end - start) * t) for t in t_values][1:]
    positions = positions[: steps // 4] + positions[len(positions) - steps // 4 :]

    return positions, start


def calculate_speeds(pan_dist, tilt_dist, idx_step, steps, max_speed=60):
    """
    Calculates pan and tilt speeds with synchronized movement.
    Uses a smooth easing function to avoid jerky transitions.
    """

    total_distance = math.sqrt(pan_dist**2 + tilt_dist**2) + 1e-6  # Avoid divide-by-zero
    pan_ratio = pan_dist / total_distance
    tilt_ratio = tilt_dist / total_distance

    # Use a smoother cubic easing function
    if idx_step < (steps // 4):
        t = idx_step / (steps // 4)
        speed = max_speed * (3 * t**2 - 2 * t**3)  # Smooth acceleration
    elif idx_step < (3 * steps // 4):
        speed = max_speed  # Maintain top speed
    else:
        t = (idx_step - (3 * steps // 4)) / (steps // 4)
        speed = max_speed - max_speed * (3 * t**2 - 2 * t**3)  # Smooth deceleration

    # Apply synchronized scaling
    pan_speed = find_closest_key(value_dict=PAN_SPEEDS, target_value=speed * pan_ratio)
    tilt_speed = find_closest_key(value_dict=TILT_SPEEDS, target_value=speed * tilt_ratio)

    return pan_speed, tilt_speed


def wait_until_position_reached(ip, dest_pan, dest_tilt, threshold=15):
    """
    Waits until the camera reaches the target pan and tilt positions within a given threshold using Euclidean distance.

    :param ip: Camera's IP address.
    :param port: Camera's VISCA port.
    :param dest_pan: Target pan position.
    :param dest_tilt: Target tilt position.
    :param threshold: Acceptable error margin for Euclidean distance.
    :param timeout: Max time to wait before assuming completion.
    """

    while True:
        time.sleep(0.05)
        ret = visca.get_camera_pan_tilt(ip)
        if ret is None:
            continue

        pan, tilt = ret

        distance = math.sqrt((dest_pan - pan) ** 2 )

        if distance <= threshold:
            return pan  # Exit when within the threshold


def move(ip, pan_pos, tilt_pos, speed, wait_for_response: bool = False) -> None:
    # fmt: off
    command = bytes(
        [
            0x81, 0x01, 0x06, 0x02,  # Command header
            speed, speed,  # Speed settings
            (pan_pos >> 12) & 0x0F, (pan_pos >> 8) & 0x0F, (pan_pos >> 4) & 0x0F, pan_pos & 0x0F,
            (tilt_pos >> 12) & 0x0F, (tilt_pos >> 8) & 0x0F, (tilt_pos >> 4) & 0x0F, tilt_pos & 0x0F,
            0xFF,  # Command terminator
        ]
    )
    # fmt: on

    visca.send_command(ip=ip, command=command, wait_for_response=wait_for_response)

def move_with_easing(ip, pan_pos, tilt_pos, steps, max_speed):
    start_pan, _ = visca.get_camera_pan_tilt(ip)  # Ignore tilt position
    pan_positions, start_pan = calculate_intermediate_positions(start_pan, pan_pos, steps)

    current_pan = start_pan
    for idx, next_pan in enumerate(pan_positions):
        pan_speed = calculate_speeds(
            pan_dist=abs(next_pan - current_pan),
            tilt_dist=0,  # No tilt movement
            idx_step=idx,
            steps=len(pan_positions),
            max_speed=max_speed,
        )[0]  # Extract only pan speed

        # Create VISCA absolute position command (only pan)
        command = bytes([
            0x81, 0x01, 0x06, 0x02,  # VISCA header
            pan_speed, pan_speed,  # Pan speed, tilt speed set to 0
            (next_pan >> 12) & 0x0F, (next_pan >> 8) & 0x0F, (next_pan >> 4) & 0x0F, next_pan & 0x0F,  # Pan position
            (tilt_pos >> 12) & 0x0F, (tilt_pos >> 8) & 0x0F, (tilt_pos >> 4) & 0x0F, tilt_pos & 0x0F,
            0xFF
        ])

        visca.send_command(ip=ip, command=command)
        #current_pan = wait_until_position_reached(ip=ip, dest_pan=next_pan, dest_tilt=None)


""" def move_with_easing(ip, pan_pos, tilt_pos, steps, max_speed):

    start_pan, start_tilt = visca.get_camera_pan_tilt(ip)
    pan_positions, start_pan = calculate_intermediate_positions(start_pan, pan_pos, steps)
    tilt_positions, start_tilt = calculate_intermediate_positions(start_tilt, tilt_pos, steps)

    current_pan, current_tilt = start_pan, start_tilt
    for idx, (next_pan, next_tilt) in enumerate(zip(pan_positions, tilt_positions)):
        pan_speed, tilt_speed = calculate_speeds(
            pan_dist=abs(next_pan - current_pan),
            tilt_dist=abs(next_tilt - current_tilt),
            idx_step=idx,
            steps=len(pan_positions),
            max_speed=max_speed,
        )

        # Create VISCA absolute position command (pan & tilt together)
        # fmt: off
        command = bytes([
            0x81, 0x01, 0x06, 0x02,  # VISCA header
            pan_speed, tilt_speed,  # Synced speeds
            (next_pan >> 12) & 0x0F, (next_pan >> 8) & 0x0F, (next_pan >> 4) & 0x0F, next_pan & 0x0F,  # Pan position
            (next_tilt >> 12) & 0x0F, (next_tilt >> 8) & 0x0F, (next_tilt >> 4) & 0x0F, next_tilt & 0x0F,  # Tilt position
            0xFF
        ])
        # fmt: on

        visca.send_command(ip=ip, command=command)
        current_pan, current_tilt = wait_until_position_reached(ip=ip, dest_pan=next_pan, dest_tilt=next_tilt) """


def hex_to_signed_int(hex_value):  
    int_value = int(hex_value, 16)
    if int_value > 0x7FFF:  # Handle two’s complement negative values
        int_value -= 0x10000
    return int_value

def visca_to_euler(hex_pan, hex_tilt):
    pan_int = hex_to_signed_int(hex_pan)
    tilt_int = hex_to_signed_int(hex_tilt)

    pan_deg = pan_int / 16.0
    tilt_deg = tilt_int / 16.0

    return pan_deg, tilt_deg

def euler_to_visca(pan_deg, tilt_deg):

    def signed_int_to_hex(value):
        if value < 0:
            value = (1 << 20) + value  # Convert to two's complement for 20-bit representation
        return f"0x{value:05X}"  # Ensure 5-digit hex format

    pan_int = int(pan_deg * 16)
    tilt_int = int(tilt_deg * 16)

    hex_pan = signed_int_to_hex(pan_int)
    hex_tilt = signed_int_to_hex(tilt_int)

    return hex_pan, hex_tilt

def calc_pan_shift(bev_x_axis_line: int, x_axis_value: int, pan_distance: float) -> float:
    result_pan = 1.0

    bev_percentage = (x_axis_value / bev_x_axis_line) * 100
    result_pan = pan_distance * (bev_percentage / 100)
    print(result_pan)
    return result_pan

def get_pan_from_bev(x_axis_value, presets):
    #get the bev-x value (horizental coordinate). between -10 to 10 

    bev_x_axis_line = 20 #
    #x_axis_value = 7 + 10 #add +10 constantly 

    pan_left_hexa = hex(presets['presets']['left'][0]) #configbol jonnek, pan left es right value of the presets
    pan_right_hexa = hex(presets['presets']['right'][0])
    tilt_hexa = hex(presets['presets']['left'][1]) #egyelore ez nem valtozik


    pan_deg_left, tilt_deg = visca_to_euler(pan_left_hexa, tilt_hexa)
    pan_deg_right, tilt_deg = visca_to_euler(pan_right_hexa, tilt_hexa)

    pan_deg_left = abs(pan_deg_left) 
    pan_deg_right = abs(pan_deg_right)

    pan_distance = pan_deg_left + pan_deg_right

    res_pan = calc_pan_shift(bev_x_axis_line, x_axis_value, pan_distance)

    pan_hex, tilt_hex = euler_to_visca(res_pan, tilt_deg)
    print(f"Pan HEX: {pan_hex}, Tilt HEX: {tilt_hex}")

    return pan_hex, tilt_hex
    # pan_hex, tilt_hexet kell elkuldeni a kameranak

def pano_process(
    config: PanoramaConfig,
    onnx_file: str,
    stop_event: Event,
    start_event: Event,
    ptz_presets,
    path,
    logger,
    bev
):
    """ """

    ffmpeg_process = subprocess.Popen(
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
            f"{config.crop[3] - config.crop[1]}x{config.crop[2] - config.crop[0]}",
            "-r",
            str(15),
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-i",
            "pipe:",
            "-c:v",
            "h264_nvenc",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            "20000k",
            "-preset",
            "fast",
            "-profile:v",
            "high",
            str(os.path.join(path, f"{Path(path).stem}_pano.mp4")),
        ],
        stdin=subprocess.PIPE,
    )

    position = 1

    sleep_time = 1 / config.fps
    bucket_width = config.frame_size[0] // 3

    onnx_session = onnxruntime.InferenceSession(onnx_file, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    
    logger.info("ONNX Model Device: %s", onnxruntime.get_device())

    window = deque()
    freq_counter = Counter()

    video_capture = cv2.VideoCapture(config.src)
    """ video_capture = cv2.VideoCapture(config.src, cv2.CAP_FFMPEG)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 3) 
    "-rtbufsize",
    "100M",
    """

    last_pos = []

    start_event.set()
    logger.info("Process Pano - Event Set!")

    ii = 0
    dist_threshold = 20

    try:
        while not stop_event.is_set():
            start_time = time.time()

            ret, frame = video_capture.read()

            if not ret:
                logger.warning(f"No panorama frame captured.")
                raise KeyboardInterrupt()

            if config.crop:
                frame = frame[config.crop[0] : config.crop[2], config.crop[1] : config.crop[3]]

            #ffmpeg_process.stdin.write(frame.tobytes())
            ffmpeg_process.stdin.flush()

            """ labels, boxes, scores = onnx_session.run(
                output_names=None,
                input_feed={
                    'images': transform(frame),
                    "orig_target_sizes": np.array([config.frame_size]),
                },
            ) """

            centroid_pos = bev.process_frame(frame, onnx_session, ii, True) if ret else print('No_Pano_Frame')
            ii = ii + 1
            centroid_pos = list(centroid_pos)

            if not None in centroid_pos and len(centroid_pos)!=0:
                if centroid_pos[0] < -dist_threshold:
                    centroid_pos[0] = -dist_threshold
                elif centroid_pos[0] > dist_threshold: 
                    centroid_pos[0] = dist_threshold

                if len(last_pos) == 0:
                    last_pos = centroid_pos
                elif abs(centroid_pos[0]-last_pos[0]) > 1 :
                    last_pos = centroid_pos

                print(ptz_presets)

                pan_hex, tilt_hex = get_pan_from_bev(last_pos[0], ptz_presets['192.168.33.101'])
                move('192.168.33.101', int(pan_hex, 16), int(tilt_hex, 16), 0x1)

                pan_hex, tilt_hex = get_pan_from_bev(-last_pos[0], ptz_presets['192.168.33.102'])
                move('192.168.33.102', int(pan_hex, 16), int(tilt_hex, 16), 0x1)
                #move_with_easing('192.168.33.101', int(pan_hex, 16), int(tilt_hex, 16), 10, 0x10)

                """ move_processes = []
                p = Process(target=move_with_easing, args=('192.168.33.101', int(pan_hex, 16), int(tilt_hex, 16), 50, 0x5))
                p.start()
                move_processes.append(p)

                for p in move_processes:
                    p.join() """

            """ most_populated_bucket = process_buckets(
                boxes=boxes,
                labels=labels,
                scores=scores,
                bucket_width=bucket_width,
            )
            mode = update_frequency(window, freq_counter, most_populated_bucket) """

            # draw(Image.fromarray(frame), labels, boxes, scores, mode, bucket_width)

            """ if position != mode:
                position = mode
                move_processes = []
                for ip, preset in ptz_presets.items():
                    pan_pos, tilt_pos = preset['presets'][num2preset[position]]
                    p = Process(target=move_with_easing, args=(ip, pan_pos, tilt_pos, 50, preset['speed']))
                    p.start()
                    move_processes.append(p)

                for p in move_processes:
                    p.join()
 """
            time.sleep(max(sleep_time - (time.time() - start_time), 0))

    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt received.")

    finally:
        video_capture.release()

        ffmpeg_process.stdin.flush()
        ffmpeg_process.stdin.close()

    logger.info("Pano Process stopped.")


class NDIReceiver:
    def __init__(
        self,
        src,
        idx: int,
        path,
        logger,
        codec: str = "h264_nvenc",
        fps: int = 30,
    ) -> None:
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
            frame = np.copy(v.data[:, :, :3])
            ndi.recv_free_video_v2(self.receiver, v)

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


def ndi_receiver_process(src, idx: int, path, stop_event: Event, logger, codec: str = "h264_nvenc", fps: int = 30):

    path = os.path.join(path, f"{Path(path).stem}_cam{idx}.mp4")
    receiver = NDIReceiver(src, idx, path, logger, codec, fps)

    logger.info(f"NDI Receiver {idx} created. Saving data to {path}")

    try:
        while not stop_event.is_set():
            frame, t = receiver.get_frame()
            if frame is not None and t == ndi.FrameType.FRAME_TYPE_VIDEO:
                try:
                    receiver.ffmpeg_process.stdin.write(frame.tobytes())
                    receiver.ffmpeg_process.stdin.flush()
                except BrokenPipeError as e:
                    logger.error(f"Broken pipe error while writing frame: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error in NDI Receiver Process {idx}: {e}")
                    break
            elif t == ndi.FrameType.FRAME_TYPE_AUDIO:
                pass
            else:
                logger.warning(f"No video frame captured. Frame type: {t}")
    except KeyboardInterrupt:
        receiver.stop()

    logger.info(f"NDI Receiver Process {receiver.idx} stopped.")


def start_cam(ip, preset) -> None:
    visca.power_on(ip)
    move(ip=ip, pan_pos=preset[0], tilt_pos=preset[1], speed=0x14)


def main(args, config: Config) -> int:
    
    bev = BEV(args)

    processes = []
    for cfg in config.camera_system.ptz_cameras.values():
        p = Process(target=start_cam, args=(cfg.ip, cfg.presets['center']))
        p.start()
        processes.append(p)

    for proc in processes:
        proc.join()

    if not ndi.initialize():
        logger.error("Failed to initialize NDI.")
        return 1

    ndi_find = ndi.find_create_v2()
    if ndi_find is None:
        logger.error("Failed to create NDI find instance.")
        return 1

    sources = []
    while len(sources) < 2:
        logger.info("Looking for sources ...")
        ndi.find_wait_for_sources(ndi_find, 5000)
        sources = ndi.find_get_current_sources(ndi_find)

    presets = {
        cfg.ip: {'presets': cfg.presets, 'speed': cfg.speed} for cfg in config.camera_system.ptz_cameras.values()
    }

    start_event = Event()
    stop_event = Event()

    proc_pano = Process(
        target=pano_process,
        args=(
            config.camera_system.pano_camera,
            config.pano_onnx,
            stop_event,
            start_event,
            presets,
            config.out_path,
            logger,
            bev
        ),
    )
    proc_pano.start()

    start_event.wait()
    processes = []
    for idx, source in enumerate(sources):
        p = Process(target=ndi_receiver_process, args=(source, idx, config.out_path, stop_event, logger))
        processes.append(p)
        p.start()

    ndi.find_destroy(ndi_find)

    try:
        delta_time = int((config.schedule.end_time - config.schedule.start_time).total_seconds())
        with tqdm(total=delta_time, bar_format="{l_bar}{bar} [Elapsed: {elapsed}, Remaining: {remaining}]") as progress:
            for _ in range(delta_time):
                time.sleep(1)
                progress.update(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Terminating processes...")

    finally:
        stop_event.set()
        for process in processes:
            if process.is_alive():
                process.join()

        proc_pano.kill()

    logger.info("Finished recording.")
    """ for cfg in config.camera_system.ptz_cameras.values():
        visca.power_off(cfg.ip) """

    return 0


if __name__ == "__main__":
    import argparse

    # multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(prog="NDI Stream Recorder", description="Schedule a script to run based on time.")
    parser.add_argument("--config", type=str, help="Config path.", required=False, default="./config.yaml")
    parser.add_argument("--start_time", type=str, required=False, default=None)
    parser.add_argument("--end_time", type=str, required=False, default=None)
    parser.add_argument("--duration", type=str, required=False, default=None)
    
    # BEV arguments
    parser.add_argument('--court-width', type=float, default=25.0, help='Width of the court in meters (default: 30)')
    parser.add_argument('--court-height', type=float, default=20.0, help='Height of the court in meters (default: 20)')
    parser.add_argument('--no-boundary', action='store_false', dest='draw_boundary', help='Disable court boundary')
    parser.add_argument('--no-half-line', action='store_false', dest='draw_half_line', help='Disable half-distance line')
    parser.add_argument('--no-2m', action='store_false', dest='draw_2m_line', help='Disable 2-meter lines')
    parser.add_argument('--no-5m', action='store_false', dest='draw_5m_line', help='Disable 5-meter lines')
    parser.add_argument('--no-6m', action='store_false', dest='draw_6m_line', help='Disable 6-meter lines')

    args = parser.parse_args()
    cfg = load_config(file_path=args.config)

    cfg.schedule.start_time = (
        datetime.strptime(args.start_time, "%Y.%m.%d_%H:%M") if args.start_time else cfg.schedule.start_time
    )
    cfg.schedule.end_time = (
        datetime.strptime(args.end_time, "%Y.%m.%d_%H:%M") if args.end_time else cfg.schedule.end_time
    )
    cfg.schedule.duration = (
        datetime.strptime(args.duration, "%Y.%m.%d_%H:%M") if args.duration else cfg.schedule.duration
    )

    logger = setup_logger(log_dir=cfg.out_path)

    schedule(cfg.schedule.start_time, cfg.schedule.end_time)

    main(args, config=cfg)
