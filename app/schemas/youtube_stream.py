from datetime import datetime

from pydantic import BaseModel


class YoutubeStreamSchedule(BaseModel):
    title: str
    description: str
    privacy_status: str
    start_time: datetime
    end_time: datetime
    category: str
