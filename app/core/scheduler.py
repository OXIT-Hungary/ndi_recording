import logging
from datetime import datetime, timezone
from threading import Event, Lock, Thread

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
        self._force_stopped = False
        self._lock = Lock()  # Add lock for thread safety

    def __str__(self):
        return f'ScheduledTask(id={self.id}, schedule={self.schedule}, task={self.task})'

    def __repr__(self):
        return str(self)

    def start(self, *args, **kwargs):
        with self._lock:
            if self.is_running() or self._force_stopped:
                return

            self.task.start(*args, **kwargs)

    def stop(self):
        with self._lock:
            if not self.is_running():
                return

            self.task.stop()

    def force_stop(self):
        with self._lock:
            self._force_stopped = True
            if self.is_running():
                self.stop()

    def is_running(self) -> bool:
        return self.task.is_running

    def is_force_stopped(self) -> bool:
        return self._force_stopped

    def is_due_to_start(self) -> bool:
        with self._lock:
            if self._force_stopped:
                return False

            now = datetime.now(timezone.utc)
            return (
                self.schedule.start_time <= now <= self.schedule.end_time
                and not self.is_running()
                and not self._force_stopped
            )

    def is_due_to_stop(self) -> bool:
        with self._lock:
            now = datetime.now(timezone.utc)
            return (self.schedule.end_time <= now or self._force_stopped) and self.is_running()


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
            tasks_to_remove = []

            for task_id, task in self.__tasks.items():
                try:
                    if task.is_due_to_stop():
                        self.__logger.debug(f"Stopping task {task_id}")
                        task.stop()
                        tasks_to_remove.append(task_id)
                    elif task.is_due_to_start():
                        self.__logger.debug(f"Starting task {task_id}")
                        task.start(task.schedule.start_time)
                except Exception as e:
                    self.__logger.error(f"Error processing task {task_id}: {e}")
                    tasks_to_remove.append(task_id)

            # Remove completed or failed tasks
            for task_id in tasks_to_remove:
                try:
                    del self.__tasks[task_id]
                except KeyError:
                    pass  # Task might have been removed by another thread

            self.__end_event.wait(self.__check_interval)

        # Cleanup remaining tasks when scheduler stops
        for task in list(self.__tasks.values()):
            try:
                task.stop()
            except Exception as e:
                self.__logger.error(f"Error stopping task during shutdown: {e}")
