from pydantic import BaseModel
from datetime import datetime


class YoutubeStreamSchedule(BaseModel):
    title: str
    description: str
    privacy_status: str
    start_time: datetime
    end_time: datetime
