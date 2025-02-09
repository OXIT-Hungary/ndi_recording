from datetime import datetime, timedelta, timezone
from typing import Annotated

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from ..core.utils.timezone import to_utc


class ScheduleMessage(BaseModel):
    id: Annotated[
        int,
        Field(
            description="ID of the schedule",
            examples=[0, 1, 2],
        ),
    ]
    remaining_time: Annotated[
        timedelta,
        Field(
            description="Remaining time until task starts",
            examples=[timedelta(days=1, hours=2, minutes=3)],
        ),
    ]
    message: Annotated[
        str,
        Field(
            description="Message of the response",
            examples=["1 day, 01:20:35", "01:20:35"],
        ),
    ]


class ScheduleRemovedMessage(BaseModel):
    id: Annotated[
        int,
        Field(
            description="ID of the schedule",
            examples=[0, 1, 2],
        ),
    ]
    message: Annotated[
        str,
        Field(
            description="Message of the response",
            examples=["Task removed"],
        ),
    ]


class ScheduledTaskIsInThePastDetailSchema(BaseModel):
    error: Annotated[
        str,
        Field(description="The error that occured", examples=["Scheduled task is in the past"]),
    ]
    scheduled_time: Annotated[
        str,
        Field(
            description="The start time of the scheduled task",
            examples=[(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()],
        ),
    ]
    current_time: Annotated[
        str,
        Field(description="The current time", examples=[datetime.now(timezone.utc).isoformat()]),
    ]


class DuplicateScheduleDetailSchema(BaseModel):
    error: Annotated[
        str,
        Field(description="The error that occured", examples=["Scheduled task with the same id already exists"]),
    ]
    id: Annotated[
        int,
        Field(description="The id of the task", examples=[0, 1, 2, 3]),
    ]


class ScheduleNotFoundDetailSchema(BaseModel):
    error: Annotated[
        str,
        Field(description="The error that occured", examples=["Scheduled task with the given id does not exist"]),
    ]
    id: Annotated[
        int,
        Field(description="The id of the task", examples=[0, 1, 2, 3]),
    ]


class OverlappingScheduleDetailSchema(BaseModel):
    error: Annotated[
        str,
        Field(description="The error that occured", examples=["Scheduled task overlaps with existing task"]),
    ]
    existing_task_id: Annotated[
        int,
        Field(description="The id of the task that overlaps", examples=[0, 1, 2, 3]),
    ]


class ScheduledTaskIsInThePastExceptionSchema(BaseModel):
    status_code: Annotated[
        int,
        Field(
            description="Status code of the exception",
            examples=[400],
        ),
    ]
    detail: Annotated[
        ScheduledTaskIsInThePastDetailSchema,
        Field(
            description="The details of the error",
        ),
    ]


class DuplicateScheduleExceptionSchema(BaseModel):
    status_code: Annotated[
        int,
        Field(
            description="Status code of the exception",
            examples=[400],
        ),
    ]
    detail: Annotated[
        DuplicateScheduleDetailSchema,
        Field(
            description="The details of the error",
        ),
    ]


class ScheduleNotFoundExceptionSchema(BaseModel):
    status_code: Annotated[
        int,
        Field(
            description="Status code of the exception",
            examples=[404],
        ),
    ]
    detail: Annotated[
        ScheduleNotFoundDetailSchema,
        Field(
            description="The details of the error",
        ),
    ]


class ScheduleOverlapsExceptionSchema(BaseModel):
    status_code: Annotated[
        int,
        Field(
            description="Status code of the exception",
            examples=[409],
        ),
    ]
    detail: Annotated[
        OverlappingScheduleDetailSchema,
        Field(
            description="The details of the error",
        ),
    ]


class Schedule(BaseModel):
    start_time: Annotated[
        datetime,
        Field(
            description="Start time of the schedule",
            examples=[(datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()],
        ),
    ]
    end_time: Annotated[
        datetime,
        Field(
            description="End time of the schedule",
            examples=[(datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()],
        ),
    ]

    def overlaps(self, other: Self) -> bool:
        return (
            self.start_time <= other.start_time
            and self.end_time >= other.start_time
            or other.start_time <= self.start_time
            and other.end_time >= self.start_time
        )


class SetSchedule(BaseModel):
    start_time: Annotated[
        datetime,
        Field(
            description="Start time of the schedule",
            examples=[
                (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat(),
                (datetime.now(timezone.utc).astimezone() + timedelta(minutes=15)).isoformat(),
            ],
        ),
    ]
    end_time: Annotated[
        datetime,
        Field(
            description="End time of the schedule",
            examples=[
                (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
                (datetime.now(timezone.utc).astimezone() + timedelta(hours=2)).isoformat(),
            ],
        ),
    ]

    @model_validator(mode="before")
    @classmethod
    def validate_schedule(cls, data: dict) -> dict:
        start_time_str: str = data["start_time"]
        end_time_str: str = data["end_time"]
        start_time: datetime = to_utc(datetime.fromisoformat(start_time_str))
        end_time: datetime = to_utc(datetime.fromisoformat(end_time_str))

        if end_time < start_time:
            raise ValueError("end_time should be greater than start_time.")

        return {"start_time": start_time, "end_time": end_time}
