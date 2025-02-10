import NDIlib as ndi
import logging
from enum import Enum
import multiprocessing

import time


from src.camera.ptz import PTZConfig, PTZCamera, PTZPosition
from src.camera.panorama import PanoramaCamera, PanoramaConfig

logger = logging.getLogger(__name__)


class CameraSystemConfig:
    def __init__(self, config_dict) -> None:
        self.ptz_cameras = {}
        self.pano_camera = None
        for cam_name, cam_config in config_dict.get("cameras", {}).items():
            if cam_name == "pano":
                self.pano_camera = PanoramaConfig(cam_config)
            elif "ptz" in cam_name:
                self.ptz_cameras[cam_name] = PTZConfig(cam_config)


class PanoProcess(multiprocessing.Process):
    """ """

    def __init__(self, config, queue, stop_event) -> None:
        super().__init__()

        self.config = config
        self.queue = queue
        self.stop_event = stop_event

        self.camera = None

        self.sleep_time = 1 / config.fps

    def run(self) -> None:
        self.camera = PanoramaCamera(config=self.config)

        while not self.stop_event.is_set():
            frame = self.camera.get_frame()
            if frame is not None:
                self.queue.put(frame)

            time.sleep(self.sleep_time)


class PTZProcess(multiprocessing.Process):
    def __init__(self, src, config, queue, stop_event) -> None:
        super().__init__()

        self.src = src
        self.config = config
        self.queue = queue
        self.stop_event = stop_event

        self.camera = None

        self.sleep_time = 1 / config.fps

    def move_to_preset(self, preset: PTZPreset, speed=0x14) -> None:
        self.camera.move_to_preset(preset=preset, speed=speed)

    def run(self) -> None:
        self.camera = PTZCamera(src=self.src, config=self.config)

        while not self.stop_event.is_set():
            frame = self.camera.get_frame()
            if frame is not None:
                self.queue.put(frame)

            time.sleep(self.sleep_time)


class CameraSystem:
    class Position(Enum):
        LEFT = 0
        CENTER = 1
        RIGHT = 2

    def __init__(self, config: CameraSystemConfig) -> None:
        self.config = config

        cams = ['pano'] if self.config.pano_camera else []
        cams += [ptz for ptz in self.config.ptz_cameras]
        self.frame_queues = {cam: multiprocessing.Queue(maxsize=10) for cam in cams}

        self.manager = multiprocessing.Manager()
        self.stop_event = self.manager.Event()
        self.panorama_process = PanoProcess(
            config=self.config.pano_camera,
            queue=self.frame_queues['pano'],
            stop_event=self.stop_event,
        )

        self.ptz_processes = self._init_ptz_cameras()
        self._position = CameraSystem.Position.CENTER

    def _init_ptz_cameras(self):
        if not ndi.initialize():
            logger.error("Failed to initialize NDI.")
            return 1

        self.ndi_find = ndi.find_create_v2()
        if self.ndi_find is None:
            logger.error("Failed to create NDI find instance.")
            return 1

        logger.info("Looking for sources ...")
        sources = []

        cnt_retry = 0
        while len(sources) < 2 and cnt_retry < 10:
            ndi.find_wait_for_sources(self.ndi_find, 8000)
            sources = ndi.find_get_current_sources(self.ndi_find)
        if cnt_retry == 10:
            logger.error("Timeout searching for NDI sources.")
            raise RuntimeError("Timeout searching for NDI sources.")

        cams = []
        for source in sources:
            cfg, cam_name = None, None
            for cam_name, cfg in self.config.ptz_cameras.items():
                if cfg.ip in source.url_address:
                    break

            cams.append(
                PTZProcess(
                    src=source,
                    config=cfg,
                    queue=self.frame_queues[cam_name],
                    stop_event=self.stop_event,
                )
            )

        for cam in cams:
            cam.move_to_preset(preset=PTZPosition.CENTER, speed=0x14)

        return cams

    def start(self) -> None:
        self.panorama_process.start()

        for ptz_process in self.ptz_processes:
            ptz_process.start()

    def stop(self) -> None:
        """Stops all camera processes gracefully."""
        self.stop_event.set()

        self.panorama_process.join()

        for process in self.ptz_processes:
            process.join()

    def get_frames(self):
        frames = {}
        for cam_id, queue in self.frame_queues.items():
            if not queue.empty():
                frame = queue.get()
                if frame is not None:
                    frames[cam_id] = frame  # Store frame

        return frames

    def move_to_preset(self, pos: Position) -> None:
        if self.position == pos:
            return

        if pos == CameraSystem.Position.LEFT:
            self.ptz_processes[0].move_to_preset(PTZPosition.LEFT)
            self.ptz_processes[1].move_to_preset(PTZPosition.RIGHT)
        elif pos == CameraSystem.Position.CENTER:
            for process in self.ptz_processes:
                process.move_to_preset(PTZPosition.CENTER)
        else:
            self.ptz_processes[0].move_to_preset(PTZPosition.RIGHT)
            self.ptz_processes[1].move_to_preset(PTZPosition.LEFT)

    @property
    def position(self) -> Position:
        return self._position

    @position.setter
    def position(self, new_pos: Position) -> None:
        self._position = new_pos

    def __del__(self) -> None:
        for process in self.ptz_processes:
            if process.is_alive():
                process.join()

        ndi.find_destroy(self.ndi_find)
