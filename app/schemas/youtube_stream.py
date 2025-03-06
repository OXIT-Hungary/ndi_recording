from pydantic import BaseModel
from datetime import datetime


class YoutubeStreamSchedule(BaseModel):
    stream_title: str
    stream_description: str
    stream_privacy_status: str = "private"  # or public or unlisted
    start_time: datetime
    end_time: datetime
