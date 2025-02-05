import time
import socket
import math
import numpy as np


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


def find_closest_key(value_dict, target_value):
    return min(value_dict, key=lambda k: abs(value_dict[k] - target_value))


def send_visca_command(ip, port, command, wait_for_response=False, timeout=6000.0):

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)
        sock.sendto(command, (ip, port))

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

        pan_position = pan_position + 65536 if pan_position < 3000 else pan_position
        tilt_position = tilt_position + 65536 if tilt_position < 3000 else tilt_position

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


def move_camera(camera_ip, camera_port, pan_pos, tilt_pos, speed):
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

    send_visca_command(camera_ip, camera_port, command, wait_for_response=True)


def move_camera_ease(ip, port, start_pan, start_tilt, end_pan, end_tilt, max_speed, steps):
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
            return pan, tilt  # Exit when within the threshold

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


def test_1(camera_ip, camera_port):
    move_camera(camera_ip=camera_ip, camera_port=camera_port, pan_pos=0xF778, tilt_pos=0x195, speed=0x14)
    time.sleep(3)

    start_pan, start_tilt = get_camera_pan_tilt(camera_ip, camera_port)
    print(f"Pan Position: {start_pan}, Tilt Position: {start_tilt}")

    END_PAN = 0x1E8  # Replace with the target pan position
    END_TILT = 0xFFF4  # Replace with the target tilt position

    MAX_SPEED = 60  # Maximum pan speed
    STEPS = 50  # Number of steps for smooth transition

    move_camera_ease(
        ip=camera_ip,
        port=camera_port,
        start_pan=start_pan,
        start_tilt=start_tilt,
        end_pan=END_PAN,
        end_tilt=END_TILT,
        max_speed=MAX_SPEED,
        steps=STEPS,
    )


def test_2(camera_ip, camera_port):
    start_pan, start_tilt = get_camera_pan_tilt(camera_ip, camera_port)
    print(f"Pan Position: {hex(start_pan & 0xFFFF)}, Tilt Position: {hex(start_tilt & 0xFFFF)}")

    for speed in range(1, 21):
        move_camera(camera_ip=camera_ip, camera_port=camera_port, pan_pos=0x3, tilt_pos=0x510, speed=0x16)
        time.sleep(3)

        start_time = time.time()
        move_camera(camera_ip=camera_ip, camera_port=camera_port, pan_pos=0x3, tilt_pos=0xFE45, speed=speed)

        print(f"Speed: {hex(speed)}\tTime: {time.time() - start_time}")


if __name__ == "__main__":
    # Camera IP and Port
    CAMERA_IP = "169.254.108.82"
    CAMERA_PORT = 52381

    test_1(CAMERA_IP, CAMERA_PORT)

    # test_2(CAMERA_IP, CAMERA_PORT)
