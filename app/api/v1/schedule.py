from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, status

from ...core.exceptions.http_exceptions import (
    DuplicateScheduleIdException,
    OverlappingScheduleException,
    ScheduledTaskIsInThePastException,
    ScheduleNotFoundException,
)
from ...core.record_manager import RecordManager
from ...core.scheduler import Scheduler, TaskNotFound, TaskOverlapsWithOtherTask, TaskWithSameIdExists
from ...core.utils.remaining_time import get_formatted_remaining_time
from ...schemas.schedule import (
    DuplicateScheduleExceptionSchema,
    Schedule,
    ScheduledTaskIsInThePastExceptionSchema,
    ScheduleMessage,
    ScheduleNotFoundExceptionSchema,
    ScheduleOverlapsExceptionSchema,
    ScheduleRemovedMessage,
)
from ...schemas.scheduled_task import ScheduledTaskSchema
from ..dependencies import get_schedule, get_scheduler

router = APIRouter(prefix="/schedule", tags=["Schedule"])


@router.get(
    "/",
    response_model=list[ScheduledTaskSchema],
    status_code=status.HTTP_200_OK,
)
def get_tasks(scheduler: Annotated[Scheduler, Depends(get_scheduler)]):
    return [
        ScheduledTaskSchema(
            id=task.id,
            schedule=task.schedule,
            is_running=task.is_running(),
            is_force_stopped=task.is_force_stopped(),
        )
        for task in scheduler.get_tasks()
    ]


@router.get(
    "/{id}",
    response_model=ScheduledTaskSchema,
    status_code=status.HTTP_200_OK,
    responses={status.HTTP_404_NOT_FOUND: {"model": ScheduleNotFoundExceptionSchema}},
)
def get_task(
    *,
    id: Annotated[int, Path(description="ID of the task to get")],
    scheduler: Annotated[Scheduler, Depends(get_scheduler)],
):
    try:
        task = scheduler.get_task(id)
        return ScheduledTaskSchema(
            id=task.id,
            schedule=task.schedule,
            is_running=task.is_running(),
            is_force_stopped=task.is_force_stopped(),
        )
    except TaskNotFound as e:
        raise ScheduleNotFoundException(id=e.id)


@router.post(
    "/",
    response_model=ScheduleMessage,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Error during setting schedule",
            "model": ScheduledTaskIsInThePastExceptionSchema,
        },
        status.HTTP_409_CONFLICT: {
            "description": "Task with same id exists or overlaps with existing task",
            "model": DuplicateScheduleExceptionSchema | ScheduleOverlapsExceptionSchema,
        },
    },
)
@router.delete(
    "/{id}",
    response_model=ScheduleRemovedMessage,
    status_code=status.HTTP_200_OK,
    responses={status.HTTP_404_NOT_FOUND: {"model": ScheduleNotFoundExceptionSchema}},
)
def remove_schedule(
    *,
    id: Annotated[int, Path(description="ID of the task to remove")],
    stop_task: Annotated[
        bool,
        Query(
            description="Stop the task if it is running",
        ),
    ] = True,
    scheduler: Annotated[Scheduler, Depends(get_scheduler)],
):
    try:
        scheduler.remove_task(id, stop_task=stop_task)
        return ScheduleRemovedMessage(id=id, message="Task removed")
    except TaskNotFound as e:
        raise ScheduleNotFoundException(id=e.id)
