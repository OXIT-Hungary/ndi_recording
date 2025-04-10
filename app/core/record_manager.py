#import logging
#import subprocess
from datetime import datetime
from threading import Lock

#import NDIlib as ndi
from multiprocess import Event, Process
#from typing_extensions import Self

#from main import ndi_receiver_process, pano_process

from .schedulable import Schedulable


class FailedToStartRecordingException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class FailedToStopRecordingException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class RecordManager(Schedulable):
    @classmethod
    def get_instance(cls, logger):
        pass

    def start(self, *args, **kwargs):
        pass

    def stop(self, *args, **kwargs):
        pass

    @property
    def is_running(self):
        pass

    def _start(self, start_time: datetime, *args, **kwargs):
        pass

    def _stop(self, *args, **kwargs):
        pass
