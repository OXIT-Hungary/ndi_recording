from multiprocessing import Manager

class StreamStatus:
    UNDEFINED = "UNDEFINED"
    STARTING = "STARTING"
    STREAMING = "STREAMING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"
    ERROR_STARTING = "ERROR_STARTING"
    ERROR_STREAMING = "ERROR_STREAMING"
    ERROR_STOPPING = "ERROR_STOPPING"

class SharedManager:
    _manager = None

    stream_status = None
    event_stop = None
    pano_queue = None

    @classmethod
    def init(cls):
        if cls._manager is None:
            cls._manager = Manager()

            cls.stream_status = cls._manager.Namespace()
            cls.event_stop = cls._manager.Event()
            cls.pano_queue = cls._manager.Queue(maxsize=5)

    @classmethod
    def get_manager(cls):
        return cls._manager


