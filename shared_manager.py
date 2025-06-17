from multiprocessing import Manager

class StreamStatus:
    UNDEFINED = "UNDEFINED"
    STARTING = "STARTING"
    STREAMING = "STREAMING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"

# Initialize a single Manager instance
_manager = Manager()

# Define shared objects here
stream_status = _manager.Namespace()
event_stop = _manager.Event()
pano_queue = _manager.Queue(maxsize=5)
