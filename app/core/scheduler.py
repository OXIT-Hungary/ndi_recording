import logging
from datetime import datetime, timezone
from threading import Event, Thread

from typing_extensions import Self

from ..schemas.schedule import Schedule
from .schedulable import Schedulable


class TaskWithSameIdExists(Exception):
    def __init__(self, message: str, id: int):
        self.message = message
        self.id = id
        super().__init__(self.message)


class TaskNotFound(Exception):
    def __init__(self, message: str, id: int):
        self.message = message
        self.id = id
        super().__init__(self.message)


class TaskOverlapsWithOtherTask(Exception):
    def __init__(self, message: str, existing_task_id: int):
        self.message = message
        self.existing_task_id = existing_task_id
        super().__init__(self.message)


class ScheduledTask:
    def __init__(self, id: int, schedule: Schedule, task: Schedulable):
        self.id = id
        self.schedule = schedule
        self.task = task
        self._running = False
        self._force_stopped = False

    def __str__(self):
        return f'ScheduledTask(id={self.id}, schedule={self.schedule}, task={self.task})'

    def __repr__(self):
        return str(self)

    def start(self, *args, **kwargs):
        if self._running:
            return

        self._running = True
        self.task.start(*args, **kwargs)

    def stop(self):
        if not self._running:
            return

        self.task.stop()
        self._running = False

    def force_stop(self):
        self._force_stopped = True
        self.stop()

    def is_running(self) -> bool:
        return self._running

    def is_force_stopped(self) -> bool:
        return self._force_stopped

    def is_due_to_start(self):
        return self.schedule.start_time <= datetime.now(timezone.utc) and not self._running and not self._force_stopped

    def is_due_to_stop(self):
        return self.schedule.end_time <= datetime.now(timezone.utc) and self._running and not self._force_stopped


class Scheduler:
    __instance: Self | None = None
    __key = object()

    @classmethod
    def get_instance(cls, logger: logging.Logger) -> Self:
        if cls.__instance is None:
            cls.__instance = cls(cls.__key, logger)
            cls.__instance.start()

        return cls.__instance

    def __init__(self, key, logger: logging.Logger, check_interval: int = 1, end_event: Event | None = None):
        if key is not self.__key:
            raise ValueError("Cannot instantiate a new instance of this class, use get_instance instead")

        self.__logger = logger

        self.__tasks: dict[int, ScheduledTask] = {}
        self.__check_interval = check_interval
        self.__end_event = end_event or Event()

        self.__thread: Thread | None = None

    def __del__(self):
        self.stop()

    def add_task(self, schedule: Schedule, task: Schedulable, id: int | None = None) -> int:
        if id is None:
            id = max(self.__tasks.keys(), default=-1) + 1

        if id in self.__tasks:
            raise TaskWithSameIdExists(f'Task with id {id} already exists', id=id)

        for existing_task in self.__tasks.values():
            if existing_task.schedule.overlaps(schedule):
                raise TaskOverlapsWithOtherTask(
                    f"Task overlaps with other task with id: {existing_task.id}", existing_task.id
                )

        self.__tasks[id] = ScheduledTask(id, schedule, task)

        return id

    def remove_task(self, id: int, stop_task: bool = True):
        if id not in self.__tasks:
            raise TaskNotFound(f'Task with id {id} does not exist', id=id)

        if stop_task:
            self.__tasks[id].stop()

        del self.__tasks[id]

    def get_task(self, id: int) -> ScheduledTask:
        if id not in self.__tasks:
            raise TaskNotFound(f'Task with id {id} does not exist', id=id)

        return self.__tasks[id]

    def get_tasks(self) -> list[ScheduledTask]:
        return list(self.__tasks.values())

    def start(self):
        self.__end_event.clear()
        self.__thread = Thread(target=self.__run, daemon=True)
        self.__thread.start()

    def stop(self):
        self.__end_event.set()
        if self.__thread is not None and self.__thread.is_alive():
            self.__thread.join()

    def stop_running_task(self) -> bool:
        for task in self.__tasks.values():
            if task.is_running():
                task.force_stop()
                return True
        return False

    def __run(self):
        while not self.__end_event.is_set():
            for task in self.__tasks.values():
                if task.is_due_to_start():
                    try:
                        task.start(task.schedule.start_time)
                    except Exception as e:
                        self.__logger.error(f"Error occured while starting a task: {e}")

                elif task.is_due_to_stop():
                    try:
                        task.stop()
                    except Exception as e:
                        self.__logger.error(f"ScheduledTask.stop encountered an error: {e}")

            self.__end_event.wait(self.__check_interval)

        for task in self.__tasks.values():
            task.stop()
