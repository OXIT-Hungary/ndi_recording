import logging

from ..core.record_manager import RecordManager
from ..core.scheduler import Scheduler
from ..core.utils.logger import get_api_logger
from ..schemas.schedule import Schedule, SetSchedule

_logger = get_api_logger()


def get_schedule(set_schedule: SetSchedule):
    return Schedule(start_time=set_schedule.start_time, end_time=set_schedule.end_time)


def get_scheduler() -> Scheduler:
    return Scheduler.get_instance(_logger)


def get_api_logger() -> logging.Logger:
    return _logger
