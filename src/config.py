import os
import yaml
from datetime import datetime, timedelta

from src.camera.camera_system import CameraSystemConfig


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


class Config:
    def __init__(self, config_dict):
        self.schedule = ScheduleConfig(config_dict=config_dict.get("schedule", {}))
        self.camera_system = CameraSystemConfig(config_dict=config_dict.get("camera_system", {}))
        self.pano_onnx = config_dict.get("pano_onnx", None)
        self.video_writer = config_dict.get("video_writer", {})

        self.out_path = f"{config_dict.get('out_path', './output')}/{datetime.now().strftime('%Y%m%d_%H%M')}"
        os.makedirs(self.out_path, exist_ok=True)


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
