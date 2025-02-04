import time
import socket
import math
import numpy as np


def send_visca_command(ip, port, command):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(1)
            sock.sendto(command, (ip, port))

    except socket.timeout:
        print("No response received from the camera.")
    except Exception as e:
        print(f"Error: {e}")


def ease_in_out_pan_tilt(ip, port, max_speed, duration):
    """
    Smooth pan/tilt with ease-in/ease-out motion.
    :param ip: Camera IP.
    :param port: Camera port.
    :param max_speed: Maximum speed (0x01 to 0x18).
    :param duration: Total duration in seconds.
    """
    # Define constants
    PAN_LEFT = 0x01  # Example direction
    TILT_UP = 0x03
    STOP = 0x03
    TERMINATOR = 0xFF

    # Ease-in
    for speed in range(1, max_speed + 1):  # Increment speed
        command = bytes([0x81, 0x01, 0x06, 0x01, speed, speed, PAN_LEFT, TILT_UP, TERMINATOR])
        send_visca_command(ip, port, command)
        time.sleep(0.05)  # Adjust timing for smoother easing

    # Maintain constant speed
    command = bytes([0x81, 0x01, 0x06, 0x01, max_speed, max_speed, PAN_LEFT, TILT_UP, TERMINATOR])
    send_visca_command(ip, port, command)
    time.sleep(duration)

    # Ease-out
    for speed in range(max_speed, 0, -1):  # Decrement speed
        print(speed)
        command = bytes([0x81, 0x01, 0x06, 0x01, speed, speed, PAN_LEFT, TILT_UP, TERMINATOR])
        send_visca_command(ip, port, command)
        time.sleep(0.05)

    # Stop
    stop_command = bytes([0x81, 0x01, 0x06, 0x01, 0x01, 0x01, STOP, STOP, TERMINATOR])
    send_visca_command(ip, port, stop_command)


def send_inquiry(ip, port, command):
    """
    Sends a VISCA inquiry command to the camera and receives the response.
    :param ip: Camera's IP address.
    :param port: Camera's port (default is usually 52381).
    :param command: The inquiry command as bytes.
    :return: The response from the camera as bytes.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(2)  # Timeout for response
        try:
            sock.sendto(command, (ip, port))  # Send the VISCA command
            response, _ = sock.recvfrom(1024)  # Receive the response
            return response
        except socket.timeout:
            raise Exception("No response from camera. Check connection or command.")
        except Exception as e:
            raise Exception(f"Error: {e}")


def get_camera_pan_tilt(ip, port):
    """
    Queries the camera for its current pan and tilt positions.
    :param ip: Camera's IP address.
    :param port: Camera's control port.
    :return: A tuple (pan_position, tilt_position).
    """
    TILT_PAN_POS_INQ = bytes.fromhex("81 09 06 12 FF")  # Inquiry command
    response = send_inquiry(ip, port, TILT_PAN_POS_INQ)

    if len(response) >= 11 and response[1] == 0x50:  # 0x50 means successful reply
        # Extract pan and tilt position values
        pan_position = (response[2] << 12) | (response[3] << 8) | (response[4] << 4) | response[5]
        tilt_position = (response[6] << 12) | (response[7] << 8) | (response[8] << 4) | response[9]
        return pan_position, tilt_position
    else:
        raise ValueError("Invalid VISCA response or no pan/tilt data.")


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
    positions = [int(start + (end - start) * t) for t in t_values]
    positions = positions[: steps // 4] + positions[len(positions) - steps // 4 :]

    # Compute distances for speed calculation
    distances = np.abs(np.diff(positions))

    positions = [pos & 0xFFFF for pos in positions]

    return positions, distances.tolist()


def calculate_speeds(pan_dist, tilt_dist, idx_step, steps, initial_speed=0x06, max_speed=0x12):
    """
    Calculates pan and tilt speeds with synchronized movement.
    Uses a smooth easing function to avoid jerky transitions.
    """
    initial_speed = max(1, min(initial_speed, max_speed))

    total_distance = math.sqrt(pan_dist**2 + tilt_dist**2) + 1e-6  # Avoid divide-by-zero
    pan_ratio = pan_dist / total_distance
    tilt_ratio = tilt_dist / total_distance

    # Use a smoother cubic easing function
    if idx_step < (steps // 4):
        t = idx_step / (steps // 4)
        speed = initial_speed + (max_speed - initial_speed) * (3 * t**2 - 2 * t**3)  # Smooth acceleration
    elif idx_step < (3 * steps // 4):
        speed = max_speed  # Maintain top speed
    else:
        t = (idx_step - (3 * steps // 4)) / (steps // 4)
        speed = max_speed - (max_speed - initial_speed) * (3 * t**2 - 2 * t**3)  # Smooth deceleration

    # Apply synchronized scaling
    pan_speed = round(speed * pan_ratio)
    tilt_speed = round(speed * tilt_ratio)

    return pan_speed, tilt_speed


def move_camera(ip, port, start_pan, start_tilt, end_pan, end_tilt, max_speed, steps):
    """
    Moves the camera from a start to an end position with ease-in and ease-out.
    :param ip: Camera's IP address.
    :param port: Camera's port.
    :param start_pan: Starting pan position.
    :param start_tilt: Starting tilt position.
    :param end_pan: Target pan position.
    :param end_tilt: Target tilt position.
    :param max_pan_speed: Maximum pan speed (0x01 to 0x18).
    :param max_tilt_speed: Maximum tilt speed (0x01 to 0x14).
    :param steps: Number of intermediate steps.
    """
    pan_positions, pan_distances = calculate_intermediate_positions(start_pan, end_pan, steps)
    tilt_positions, tilt_distances = calculate_intermediate_positions(start_tilt, end_tilt, steps)

    for idx, (pan_dist, tilt_dist) in enumerate(zip(pan_distances, tilt_distances)):
        pan_speed, tilt_speed = calculate_speeds(
            pan_dist=pan_dist,
            tilt_dist=tilt_dist,
            idx_step=idx,
            steps=len(pan_positions),
            max_speed=max_speed,
        )

        print(pan_speed)

        pan_pos = pan_positions[idx]
        tilt_pos = tilt_positions[idx]

        # Create VISCA absolute position command (pan & tilt together)
        # fmt: off
        command = bytes([
            0x81, 0x01, 0x06, 0x02,  # VISCA header
            pan_speed, tilt_speed,  # Synced speeds
            (pan_pos >> 12) & 0x0F, (pan_pos >> 8) & 0x0F, (pan_pos >> 4) & 0x0F, pan_pos & 0x0F,  # Pan position
            (tilt_pos >> 12) & 0x0F, (tilt_pos >> 8) & 0x0F, (tilt_pos >> 4) & 0x0F, tilt_pos & 0x0F,  # Tilt position
            0xFF
        ])
        # fmt: on

        send_visca_command(ip, port, command)
        wait_until_position_reached(ip, port, pan_pos, tilt_pos)


def wait_until_position_reached(ip, port, dest_pan, dest_tilt, threshold=15):
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
            break  # Exit when within the threshold

        # time.sleep(0.05)  # Adjust polling interval for smoother checking


def get_camera_position(ip, port):
    """
    Sends VISCA Inquiry to get the camera's current pan/tilt position.
    """
    TILT_PAN_POS_INQ = bytes.fromhex("81 09 06 12 FF")  # Inquiry command
    response = send_inquiry(ip, port, TILT_PAN_POS_INQ)

    if len(response) >= 11 and response[1] == 0x50:  # 0x50 means successful reply
        # Extract pan and tilt position values
        pan_position = (response[2] << 12) | (response[3] << 8) | (response[4] << 4) | response[5]
        tilt_position = (response[6] << 12) | (response[7] << 8) | (response[8] << 4) | response[9]
        return pan_position, tilt_position
    else:
        raise ValueError("Invalid VISCA response or no pan/tilt data.")


if __name__ == "__main__":
    # Camera IP and Port
    CAMERA_IP = "169.254.108.82"
    CAMERA_PORT = 52381

    pan_pos = 0xF778
    tilt_pos = 0x195

    # fmt: off
    command = bytes(
        [
            0x81, 0x01, 0x06, 0x02,  # Command header
            0x16, 0x16,  # Speed settings
            (pan_pos >> 12) & 0x0F, (pan_pos >> 8) & 0x0F, (pan_pos >> 4) & 0x0F, pan_pos & 0x0F,
            (tilt_pos >> 12) & 0x0F, (tilt_pos >> 8) & 0x0F, (tilt_pos >> 4) & 0x0F, tilt_pos & 0x0F,
            0xFF,  # Command terminator
        ]
    )
    # fmt: on

    send_visca_command(CAMERA_IP, CAMERA_PORT, command)
    time.sleep(3)

    # ----------------------
    # Cam Movement 1 Test
    # ----------------------

    start_pan, start_tilt = get_camera_pan_tilt(CAMERA_IP, CAMERA_PORT)
    print(f"Pan Position: {start_pan}, Tilt Position: {start_tilt}")

    END_PAN = 0x1E8  # Replace with the target pan position
    END_TILT = 0xFFF4  # Replace with the target tilt position

    MAX_SPEED = 0x12  # Maximum pan speed
    STEPS = 50  # Number of steps for smooth transition

    move_camera(
        ip=CAMERA_IP,
        port=CAMERA_PORT,
        start_pan=start_pan,
        start_tilt=start_tilt,
        end_pan=END_PAN,
        end_tilt=END_TILT,
        max_speed=MAX_SPEED,
        steps=STEPS,
    )

    # ----------------------
    # Cam Movement 2 Test
    # ----------------------

    # # fmt: off
    # command = bytes(
    #     [
    #         0x81, 0x01, 0x06, 0x02,  # Command header
    #         0x16, 0x16,  # Speed settings
    #         (pan_pos >> 12) & 0x0F, (pan_pos >> 8) & 0x0F, (pan_pos >> 4) & 0x0F, pan_pos & 0x0F,
    #         (tilt_pos >> 12) & 0x0F, (tilt_pos >> 8) & 0x0F, (tilt_pos >> 4) & 0x0F, tilt_pos & 0x0F,
    #         0xFF,  # Command terminator
    #     ]
    # )
    # # fmt: on

    # send_visca_command(CAMERA_IP, CAMERA_PORT, command)

    # CLEAR_ALL_COMMAND = bytes.fromhex("81 01 00 01 FF")  # Clear All Command
    # POWER_ON_COMMAND = bytes.fromhex("81 01 04 00 02 FF")  # Power On Command

    # # Step 1: Clear All
    # print("Sending Clear All Command...")
    # send_visca_command(CAMERA_IP, CAMERA_PORT, CLEAR_ALL_COMMAND)

    # # Step 2: Attempt Power On
    # print("Attempting to Power On the Camera...")
    # send_visca_command(CAMERA_IP, CAMERA_PORT, POWER_ON_COMMAND)
