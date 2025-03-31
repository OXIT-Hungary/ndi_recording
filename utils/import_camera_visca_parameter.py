import socket

VISCA_PORT = 52381
ip = "192.168.33.102"

def get_camera_pan_tilt(ip):
    """
    Queries the camera for its current pan and tilt positions.
    :param ip: Camera's IP address.
    :return: A tuple (pan_position, tilt_position).
    """

    response = send_inquiry(ip, command=bytes.fromhex("81 09 06 12 FF"))
    #print(response)
    if len(response) >= 11 and response[1] == 0x50:  # 0x50 means successful reply
        # Extract pan and tilt position values
        pan_position = (response[2] << 12) | (response[3] << 8) | (response[4] << 4) | response[5]
        tilt_position = (response[6] << 12) | (response[7] << 8) | (response[8] << 4) | response[9]

        pan_position = pan_position + 65536 if pan_position < 3000 else pan_position
        #print(hex(tilt_position))
        tilt_position = tilt_position + 65536 if tilt_position < 3000 else tilt_position

        return hex(pan_position), hex(tilt_position)
    else:
        return None



def send_inquiry(ip, command, timeout: float = 10.0):
    """
    Sends a VISCA inquiry command to the camera and receives the response.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)  # Timeout for response
        try:
            sock.sendto(command, (ip, VISCA_PORT))  # Send the VISCA command
            response, _ = sock.recvfrom(1024)  # Receive the response
            return response
        except TimeoutError as e:
            raise TimeoutError("[ERROR] No response from camera. Camera IP: %s", ip) from e
        except Exception as e:
            raise Exception(f"Error: {e}") from e

def hex_to_signed_int(hex_value):
    """Converts a 16-bit hexadecimal VISCA value to a signed integer."""
    int_value = int(hex_value, 16)
    if int_value > 0x7FFF:  # Handle two’s complement negative values
        int_value -= 0x10000
    return int_value


def visca_to_euler(hex_pan, hex_tilt):
    """
    Converts VISCA hexadecimal pan-tilt values to Euler degrees.
    """
    pan_int = hex_to_signed_int(hex_pan)
    tilt_int = hex_to_signed_int(hex_tilt)

    pan_deg = pan_int / 16.0
    tilt_deg = tilt_int / 16.0

    return pan_deg, tilt_deg

print(get_camera_pan_tilt(ip))

p, t = get_camera_pan_tilt(ip)
pan_deg, tilt_deg = visca_to_euler(p, t)
print(f"Pan: {pan_deg}°, Tilt: {tilt_deg}°")