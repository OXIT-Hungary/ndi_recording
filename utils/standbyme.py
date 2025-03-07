import time
import socket
import math

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
    

CAMERA_IP = "192.168.33.101"
CAMERA_PORT = 52381

send_visca_command(CAMERA_IP, CAMERA_PORT, bytes.fromhex("81 0A 01 06 01 FF"))
 