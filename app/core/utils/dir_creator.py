import os
from datetime import datetime

API_DIR = f"{os.getcwd()}/output/api"
RECORDING_DIR = f"{os.getcwd()}/output"
os.makedirs(API_DIR, exist_ok=True)
os.makedirs(RECORDING_DIR, exist_ok=True)


def get_api_dir() -> str:
    return API_DIR


def get_recording_dir_from_date_str(date_str: str) -> str:
    recording_dir = f"{RECORDING_DIR}/{date_str}"
    os.makedirs(recording_dir, exist_ok=True)
    return recording_dir


def get_recording_dir_from_datetime(dt: datetime) -> str:
    date_str = f"{dt.astimezone().strftime('%Y%m%d_%H%M')}"
    return get_recording_dir_from_date_str(date_str)
