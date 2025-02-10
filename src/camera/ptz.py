import NDIlib as ndi
import numpy as np
import socket
import time
import math
import logging
from enum import Enum

logger = logging.getLogger(__name__)

VISCA_PORT = 52381


class PTZConfig:
    def __init__(self, ptz_dict):
        self.name = ptz_dict.get("name", None)
        self.enable = ptz_dict.get("enable", False)
        self.ip = ptz_dict.get("ip", None)
        self.resolution = ptz_dict.get("resolution", [1920, 1080])
        self.codec = ptz_dict.get("codec", "h264_nvenc")
        self.ext = ptz_dict.get("ext", ".mp4")
        self.fps = ptz_dict.get("fps", 30)
        self.bitrate = ptz_dict.get("bitrate", 40000)
        self.presets = ptz_dict.get("presets", None)


class PTZPosition(Enum):
    LEFT = 'left'
    CENTER = 'center'
    RIGHT = 'right'


class PTZCamera:
    """PTZ Camera class."""

    def __init__(self, src, config: PTZConfig) -> None:

        self.config = config
        self.receiver = self._create_receiver(src)

        self.power_on()
        self._position = None

    def _create_receiver(self, src):

        ndi_recv_create = ndi.RecvCreateV3()
        ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
        receiver = ndi.recv_create_v3(ndi_recv_create)
        if receiver is None:
            logger.error(f"Failed to create NDI receiver. Camera IP: {self.config.ip}")
            raise RuntimeError(f"Failed to create NDI receiver. Camera IP: {self.config.ip}")
        ndi.recv_connect(receiver, src)

        return receiver

    def _send_inquiry(self, command, timeout: float = 2.0):
        """
        Sends a VISCA inquiry command to the camera and receives the response.
        :param ip: Camera's IP address.
        :param port: Camera's port (default is usually 52381).
        :param command: The inquiry command as bytes.
        :return: The response from the camera as bytes.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(timeout)  # Timeout for response
            try:
                sock.sendto(command, (self.config.ip, VISCA_PORT))  # Send the VISCA command
                response, _ = sock.recvfrom(1024)  # Receive the response
                return response
            except socket.timeout:
                logger.error(f"No response from camera. Camera IP: {self.config.ip}")
                raise Exception(f"No response from camera. Camera IP: {self.config.ip}")
            except Exception as e:
                logger.error(f"Error: {e}")
                raise Exception(f"Error: {e}")

    def _send_visca_command(self, command, wait_for_response: bool = False, timeout: float = 6000.0):
        """ """
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(timeout)
            sock.sendto(command, (self.config.ip, VISCA_PORT))

            if not wait_for_response:
                return True

            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response, _ = sock.recvfrom(1024)
                    if response == b'\x90\x41\xff':  # ACK (Command Received)
                        pass
                    elif response == b'\x90\x51\xff':  # Completion (Command Executed)
                        return True
                except socket.timeout:
                    print("Timeout waiting for VISCA response.")
                    return False

            print("No completion response received within timeout.")
            return False

    def _calculate_intermediate_positions(self, start, end, steps):
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

    def _calculate_speeds(self, pan_dist, tilt_dist, idx_step, steps, max_speed=60):
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

    def wait_until_position_reached(self, dest_pan, dest_tilt, threshold=15):
        """
        Waits until the camera reaches the target pan and tilt positions within a given threshold using Euclidean distance.

        :param ip: Camera's IP address.
        :param port: Camera's VISCA port.
        :param dest_pan: Target pan position.
        :param dest_tilt: Target tilt position.
        :param threshold: Acceptable error margin for Euclidean distance.
        :param timeout: Max time to wait before assuming completion.
        """

        time.sleep(0.05)  # Small delay to allow initial movement

        while True:
            pan, tilt = get_camera_pan_tilt(ip, port)  # Get current camera position

            distance = math.sqrt((dest_pan - pan) ** 2 + (dest_tilt - tilt) ** 2)

            if distance <= threshold:
                return pan, tilt  # Exit when within the threshold

            # time.sleep(0.05)  # Adjust polling interval for smoother checking

    def get_camera_pan_tilt(self):
        """
        Queries the camera for its current pan and tilt positions.

        :return: A tuple (pan_position, tilt_position).
        """
        TILT_PAN_POS_INQ = bytes.fromhex("81 09 06 12 FF")  # Inquiry command
        response = self.__send_inquiry(TILT_PAN_POS_INQ)

        if len(response) >= 11 and response[1] == 0x50:  # 0x50 means successful reply
            # Extract pan and tilt position values
            pan_position = (response[2] << 12) | (response[3] << 8) | (response[4] << 4) | response[5]
            tilt_position = (response[6] << 12) | (response[7] << 8) | (response[8] << 4) | response[9]

            pan_position = pan_position + 65536 if pan_position < 3000 else pan_position
            tilt_position = tilt_position + 65536 if tilt_position < 3000 else tilt_position

            return pan_position, tilt_position
        else:
            raise ValueError("Invalid VISCA response or no pan/tilt data.")

    def get_frame(self):

        t, v, _, _ = ndi.recv_capture_v3(self.receiver, 1000)
        frame = None
        if t == ndi.FRAME_TYPE_VIDEO:
            frame = np.copy(v.data[:, :, :3])
            ndi.recv_free_video_v2(self.receiver, v)

        return frame, t

    def power_on(self) -> None:
        self._send_visca_command(command=bytes.fromhex("81 01 00 01 FF"), wait_for_response=True)
        self._send_visca_command(command=bytes.fromhex("81 01 04 00 02 FF"), wait_for_response=True)

    def power_off(self) -> None:
        self._send_visca_command(command=bytes.fromhex("81 01 00 01 FF"), wait_for_response=True)
        self._send_visca_command(command=bytes.fromhex("81 01 04 00 03 FF"), wait_for_response=True)

    def move(self, pan_pos, tilt_pos, speed, wait_for_response: bool = False) -> None:
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

        self._send_visca_command(command, wait_for_response=wait_for_response)

    def move_with_easing(self):
        pan_positions, start_pan = calculate_intermediate_positions(start_pan, end_pan, steps)
        tilt_positions, start_tilt = calculate_intermediate_positions(start_tilt, end_tilt, steps)

        current_pan, current_tilt = start_pan, start_tilt
        for idx, (next_pan, next_tilt) in enumerate(zip(pan_positions, tilt_positions)):
            pan_speed, tilt_speed = calculate_speeds(
                pan_dist=abs(next_pan - current_pan),
                tilt_dist=abs(next_tilt - current_tilt),
                idx_step=idx,
                steps=len(pan_positions),
                max_speed=max_speed,
            )

            print(pan_speed)

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

            send_visca_command(ip, port, command)
            current_pan, current_tilt = wait_until_position_reached(ip, port, next_pan, next_tilt)

    def move_to_preset(self, pos: PTZPosition, speed, wait_for_response=False) -> None:
        pan_pos, tilt_pos = self.config.presets[pos.value]
        self.move(pan_pos=pan_pos, tilt_pos=tilt_pos, speed=speed, wait_for_response=wait_for_response)

        self._position = pos
