from multiprocessing import Manager, Value
from enum import IntEnum

# class StreamStatus:
#     UNDEFINED = "UNDEFINED"
#     STARTING = "STARTING"
#     STREAMING = "STREAMING"
#     STOPPING = "STOPPING"
#     STOPPED = "STOPPED"
#     ERROR = "ERROR"
#     ERROR_STARTING = "ERROR_STARTING"
#     ERROR_STREAMING = "ERROR_STREAMING"
#     ERROR_STOPPING = "ERROR_STOPPING"

class StreamStatus(IntEnum):
    UNDEFINED = 0
    STARTING = 1
    STREAMING = 2
    STOPPING = 3
    STOPPED = 4
    ERROR = 5
    ERROR_STARTING = 6
    ERROR_STREAMING = 7
    ERROR_STOPPING = 8

class SharedManager:
    _manager = None

    stream_status = None
    event_stop = None
    pano_queue = None

    @classmethod
    def init(cls):
        if cls._manager is None:
            cls._manager = Manager()

            cls.stream_status = cls._manager.Value('i', StreamStatus.UNDEFINED)
            cls.event_stop = cls._manager.Event()
            cls.pano_queue = cls._manager.Queue(maxsize=5)

    @classmethod
    def get_manager(cls):
        return cls._manager


