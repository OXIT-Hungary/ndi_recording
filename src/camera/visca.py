import logging
import socket
import time

logger = logging.getLogger(__name__)


def power_on(ip, port: int = 52381) -> None:
    response = send_inquiry(ip=ip, command=bytes.fromhex("81 09 04 00 FF"), port=port)
    if response[2] == 0x03:  # Camera off
        send_command(ip=ip, command=bytes.fromhex("81 01 00 01 FF"), wait_for_response=False, port=port)
        send_command(ip=ip, command=bytes.fromhex("81 01 04 00 02 FF"), wait_for_response=True, timeout=15, port=port)


def power_off(ip, port: int = 52381) -> None:
    response = send_inquiry(ip=ip, command=bytes.fromhex("81 09 04 00 FF"), port=port)
    if response[2] == 0x02:  # Camera off
        send_command(ip=ip, command=bytes.fromhex("81 01 00 01 FF"), wait_for_response=False, port=port)
        send_command(ip=ip, command=bytes.fromhex("81 01 04 00 03 FF"), wait_for_response=True, timeout=15, port=port)


def send_inquiry(ip, command, timeout: float = 0.5, port: int = 52381):
    """
    Sends a VISCA inquiry command to the camera and receives the response.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)  # Timeout for response
        try:
            sock.sendto(command, (ip, port))  # Send the VISCA command
            response, _ = sock.recvfrom(1024)  # Receive the response
            return response
        except TimeoutError as e:
            logger.error(e)
        except Exception as e:
            logger.error("Error: %s", e)
            raise e


def send_command(ip, command, wait_for_response: bool = False, timeout: float = 2.0, port: int = 52381):
    """ """
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
            except TimeoutError:
                logger.debug("Timeout waiting for VISCA response.")
                return False

        logger.debug("No completion response received within timeout.")
        return False


def get_camera_pan_tilt(ip, port: int = 52381) -> tuple[int, int] | None:
    """
    Queries the camera for its current pan and tilt positions.
    :param ip: Camera's IP address.
    :return: A tuple (pan_position, tilt_position).
    """

    response = send_inquiry(ip, command=bytes.fromhex("81 09 06 12 FF"), port=port)

    if response is not None and len(response) >= 11 and response[1] == 0x50:  # 0x50 means successful reply
        # Extract pan and tilt position values
        pan_position = (response[2] << 12) | (response[3] << 8) | (response[4] << 4) | response[5]
        tilt_position = (response[6] << 12) | (response[7] << 8) | (response[8] << 4) | response[9]

        pan_position = pan_position + 65536 if pan_position < 3000 else pan_position
        tilt_position = tilt_position + 65536 if tilt_position < 3000 else tilt_position

        return pan_position, tilt_position
    else:
        return None, None
