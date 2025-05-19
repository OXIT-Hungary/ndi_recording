import os
from datetime import datetime, timedelta

import yaml
import numpy as np


class ScheduleConfig:
    def __init__(self, config_dict):
        self.start_time = config_dict.get("start_time")
        self.end_time = config_dict.get("end_time")
        self.duration = config_dict.get("duration")

        self.calculate()

    def calculate(self) -> None:
        now = datetime.now()

        if self.start_time:
            splt = self.start_time.split("_")
            if len(splt) == 1:
                h, m = splt[0].split(":")
                self.start_time = datetime.strptime(f"{now.year}.{now.month}.{now.day}_{h}:{m}", "%Y.%m.%d_%H:%M")
            else:
                self.start_time = datetime.strptime(self.start_time, "%Y.%m.%d_%H:%M")
        else:
            self.start_time = now

        if self.end_time:
            splt = self.end_time.split("_")
            if len(splt) == 1:
                h, m = splt[0].split(":")
                self.end_time = datetime.strptime(f"{now.year}.{now.month}.{now.day}_{h}:{m}", "%Y.%m.%d_%H:%M")
            else:
                self.end_time = datetime.strptime(self.end_time, "%Y.%m.%d_%H:%M")

        elif self.duration:
            self.duration = datetime.strptime(self.duration, "%H:%M").time()
            self.end_time = self.start_time + timedelta(hours=self.duration.hour, minutes=self.duration.minute)

        else:
            self.duration = datetime.strptime("01:45", "%H:%M").time()
            self.end_time = self.start_time + timedelta(hours=self.duration.hour, minutes=self.duration.minute)


class PTZConfig:
    def __init__(self, config_dict):
        self.name = config_dict.get("name", None)
        self.enable = config_dict.get("enable", False)
        self.ip = config_dict.get("ip", None)
        self.visca_port = config_dict.get("visca_port", 1259)
        self.resolution = config_dict.get("resolution", [1920, 1080])
        self.codec = config_dict.get("codec", "h264_nvenc")
        self.ext = config_dict.get("ext", ".mp4")
        self.fps = config_dict.get("fps", 30)
        self.bitrate = config_dict.get("bitrate", 40000)
        self.presets = config_dict.get("presets", None)
        self.speed = config_dict.get("speed", 0x10)
        self.stream = config_dict.get("stream", False)


class PanoramaConfig:
    def __init__(self, config_dict):
        self.enable = config_dict.get("enable", False)
        self.src = config_dict.get("src")
        self.crop = config_dict.get("crop", None)
        self.fps = config_dict.get("fps", 15)
        self.save = config_dict.get("save", True)


class CameraSystemConfig:
    def __init__(self, config_dict) -> None:
        self.ptz_cameras = {}
        self.pano_camera = None

        for cam_name, cam_config in config_dict.get("cameras", {}).items():
            if cam_name == "pano":
                self.pano_camera = PanoramaConfig(cam_config)
            elif "ptz" in cam_name:
                self.ptz_cameras[cam_name] = PTZConfig(cam_config)

        self.pano_onnx = config_dict.get("pano_onnx", None)
        self.track_threshold = config_dict.get("track_threshold", 1.0)


class BEVConfig:
    def __init__(self, config_dict):
        self.points = {key: np.array(val) for key, val in config_dict.get("points", {}).items()}
        self.court_size = np.array(config_dict.get("court_size", [25, 20]))
        self.court_padding = np.array(config_dict.get("court_padding", [2, 1]))


class Config:
    def __init__(self, config_dict):
        self.schedule = ScheduleConfig(config_dict=config_dict.get("schedule", {}))
        self.camera_system = CameraSystemConfig(config_dict=config_dict.get("camera_system", {}))
        self.bev = BEVConfig(config_dict=config_dict.get("bev", {}))

        self.out_path = f"{config_dict.get('out_path', './output')}"


def load_config(file_path: str):
    """Loads YAML file and returns a Config object."""
    try:
        with open(file_path, 'r') as file:
            return Config(yaml.safe_load(file) or {})
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return Config({})
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return Config({})
