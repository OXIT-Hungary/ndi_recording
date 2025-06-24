import math
import multiprocessing
import os
import subprocess
import threading
import time
from pathlib import Path
import struct

import NDIlib as ndi
import numpy as np

import src.camera.visca as visca
from src.camera.camera import Camera
from src.config import PTZConfig

NUM2PRESET = {0: 'left', 1: 'center', 2: 'right'}
PRESET2NUM = {'left': 0, 'center': 1, 'right': 2}


class PTZCamera(Camera, multiprocessing.Process):
    """PTZ Camera class."""

    PAN_SPEEDS = None
    TILT_SPEEDS = None

    def __init__(
        self,
        name: str,
        config: PTZConfig,
        event_stop: multiprocessing.Event,
        out_path: str,
        queue_move: multiprocessing.Queue,
        event_move: multiprocessing.Event,
        stream_token: str = None,
    ) -> None:
        Camera.__init__(self, event_stop=event_stop)
        multiprocessing.Process.__init__(self)

        self.name = name
        self.config = config
        self.out_path = out_path
        self.receiver = None

        self.ip = config.ip
        self.visca_port = config.visca_port
        self.presets = config.presets
        self.sleep_time = 1 / config.fps

        self.queue_move = queue_move
        self._event_move = event_move
        self._thread_move = None

        self.ffmpeg = None
        self.ffmpeg_stream = None
        self.stream_token = stream_token

    def _create_receiver(self):

        sources = self._init_ndi()
        src = next((s for s in sources if self.ip in s.url_address), None)

        ndi_recv_create = ndi.RecvCreateV3()
        ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
        receiver = ndi.recv_create_v3(ndi_recv_create)
        if receiver is None:
            raise RuntimeError("Failed to create NDI receiver")
        ndi.recv_connect(receiver, src)

        return receiver

    def _init_ndi(self) -> list:
        if not ndi.initialize():
            logger.error("Failed to initialize NDI.")
            return 1

        self._ndi_find = ndi.find_create_v2()
        if self._ndi_find is None:
            logger.error("Failed to create NDI find instance.")
            return 1

        sources = []
        while len(sources) < 2:
            # logger.info("Looking for sources ...")
            ndi.find_wait_for_sources(self._ndi_find, 5000)
            sources = ndi.find_get_current_sources(self._ndi_find)

        return sources

    def run(self) -> None:
        self._thread_move = threading.Thread(target=self._move_thread, daemon=True)
        self._thread_move.start()

        try:
            self.receiver = self._create_receiver()
            # fmt: off
            self.ffmpeg = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel", "error",
                        "-f", "rawvideo",
                        "-pix_fmt", "bgr24",
                        "-s", "1920x1080",
                        "-r", str(self.config.fps),
                        "-hwaccel", "cuda",
                        "-hwaccel_output_format", "cuda",
                        "-i", "pipe:",
                        "-c:v", self.config.codec,
                        "-pix_fmt", "yuv420p",
                        "-b:v", f"{self.config.bitrate}k",
                        "-preset", "fast",
                        "-profile:v", "high",
                        os.path.join(self.out_path, f"{Path(self.out_path).stem}_{self.name}.mp4"),
                    ],
                    stdin=subprocess.PIPE,
                )
            
            if self.config.stream:
                self.ffmpeg_stream = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-loglevel", "error",
                        # Video input (from stdin)
                        "-f", "rawvideo",
                        "-pix_fmt", "bgr24",
                        "-s", "1920x1080",
                        "-r", str(self.config.fps),
                        "-i", "-",
                        # Audio input (generate silence)
                        "-f", "lavfi",
                        "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                        # Video encoding
                        "-vcodec", "h264_nvenc",
                        "-preset", "fast",
                        "-b:v", "4500k",
                        # Audio encoding
                        "-c:a", "aac",
                        "-ar", "44100",
                        "-b:a", "128k",
                        # Output format
                        "-f", "flv",
                        f"rtmp://a.rtmp.youtube.com/live2/{self.stream_token}"
                    ],
                stdin=subprocess.PIPE,
                )
            # fmt: on

            #f = open(os.path.join(self.out_path, f"{Path(self.out_path).stem}_{self.name}.bin"), "wb")

            while not self.event_stop.is_set():
                start_time = time.time()
                frame = self.get_frame()
                if frame is not None:
                    self.ffmpeg.stdin.write(frame.tobytes())
                    self.ffmpeg.stdin.flush()

                    if self.ffmpeg_stream is not None:
                        self.ffmpeg_stream.stdin.write(frame.tobytes())
                        self.ffmpeg_stream.stdin.flush()

                    pan, tilt = visca.get_camera_pan_tilt(ip=self.ip, port=self.visca_port)
                    #f.write(struct.pack('ii', pan, tilt))

                time.sleep(max(self.sleep_time - (time.time() - start_time), 0))
        except Exception as e:
            print(f"PTZ Camera: {e}")

        finally:
            if self.ffmpeg is not None:
                self.ffmpeg.stdin.flush()
                self.ffmpeg.stdin.close()

            if self.ffmpeg_stream is not None:
                self.ffmpeg_stream.stdin.flush()
                self.ffmpeg_stream.stdin.close()

            #if not f.closed:
             #   f.close()

    def get_frame(self) -> np.ndarray | None:
        """ """

        t, v, _, _ = ndi.recv_capture_v3(self.receiver, 1000)
        frame = None
        if t == ndi.FRAME_TYPE_VIDEO:
            frame = np.copy(v.data[:, :, :3])
            ndi.recv_free_video_v2(self.receiver, v)

        return frame

    def power_on(self) -> None:
        """Powers on camera with VISCA command."""

        visca.power_on(ip=self.ip, port=self.visca_port)
        self.move_to_preset(preset='center', speed=0x14)

    def power_off(self) -> None:
        """Powers off camera with VISCA command."""

        visca.power_off(ip=self.ip, port=self.visca_port)

    def move_with_easing(self, pan_pos, tilt_pos, steps, max_speed):
        """
        Moves camera with ease-in ease-out to a specified pan-tilt position.

        Args:
            pan_pos (int):
            tilt_pos (int):
            speed (int):
        """

        start_pan, start_tilt = visca.get_camera_pan_tilt(ip=self.ip, port=self.visca_port)
        pan_positions, start_pan = self._calculate_intermediate_positions(start_pan, pan_pos, steps)
        tilt_positions, start_tilt = self._calculate_intermediate_positions(start_tilt, tilt_pos, steps)

        current_pan, current_tilt = start_pan, start_tilt
        for idx, (next_pan, next_tilt) in enumerate(zip(pan_positions, tilt_positions)):
            pan_speed, tilt_speed = self._calculate_speeds(
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

            visca.send_command(ip=self.ip, command=command, port=self.visca_port)
            current_pan, current_tilt = self._wait_until_position_reached(dest_pan=next_pan, dest_tilt=next_tilt)

    def move(
        self, ip, visca_port: int, pan_pos: int, tilt_pos: int, speed: int, wait_for_response: bool = False
    ) -> None:
        """
        Moves camera to a specified pan-tilt position.

        Args:
            pan_pos (int):
            tilt_pos (int):
            speed (int):
        """

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

        visca.send_command(ip=ip, command=command, port=visca_port, wait_for_response=wait_for_response)

    def _calculate_intermediate_positions(self, start, end, steps) -> tuple[list, int]:
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

    def _calculate_speeds(
        self,
        pan_dist: int,
        tilt_dist: int,
        idx_step: int,
        steps: int,
        max_speed: int = 60,
    ) -> tuple[int, int]:
        """
        Calculates pan and tilt speeds with synchronized movement.
        Uses a smooth easing function to avoid jerky transitions.

        Args:
            pan_dist (int):
            tilt_dist (int):
            idx_step (int):
            steps (int):
            max_speed (int):

        Returns: tuple[int, int]
            - Pan speed
            - Tilt speed
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
        pan_speed = self._find_closest_key(value_dict=self.PAN_SPEEDS, target_value=speed * pan_ratio)
        tilt_speed = self._find_closest_key(value_dict=self.TILT_SPEEDS, target_value=speed * tilt_ratio)

        return pan_speed, tilt_speed

    def _find_closest_key(self, value_dict, target_value):
        return min(value_dict, key=lambda k: abs(value_dict[k] - target_value))

    def _wait_until_position_reached(self, dest_pan: int, dest_tilt: int, threshold: int = 5) -> tuple[int, int]:
        """
        Waits until the camera reaches the target pan and tilt positions within a given threshold using Euclidean distance.

        Args:
            dest_pan (int): Target pan position.
            dest_tilt (int): Target tilt position.
            threshold (int): Acceptable error margin for Euclidean distance.
            timeout (int): Max time to wait before assuming completion.

        Returns: tuple[int, int]
            - Pan position
            - Tilt position
        """

        while not self.event_stop.is_set():
            time.sleep(0.05)
            ret = visca.get_camera_pan_tilt(ip=self.ip, port=self.visca_port)
            if ret is None:
                continue

            pan, tilt = ret

            distance = math.sqrt((dest_pan - pan) ** 2 + (dest_tilt - tilt) ** 2)

            if distance <= threshold:
                return pan, tilt  # Exit when within the threshold

    def _move_thread(self) -> None:
        while not self.event_stop.is_set():
            try:
                self._event_move.wait()

                if self.event_stop.is_set():
                    break

                if not self.queue_move.empty():
                    pan_pos, tilt_pos = self.queue_move.get()

                    self.move_with_easing(pan_pos=pan_pos, tilt_pos=65158, steps=20, max_speed=self.config.speed)

                self._event_move.clear()
            except Exception as e:
                print(e)


class Avonic_CM93_NDI(PTZCamera):
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

    def __init__(self, name, config, event_stop, out_path, queue_move, event_move, stream_token):
        super().__init__(name, config, event_stop, out_path, queue_move, event_move, stream_token)
