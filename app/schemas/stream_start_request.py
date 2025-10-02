from pydantic import BaseModel, Field
from typing import Optional

class StreamStartRequest(BaseModel):
    division: Optional[str] = Field(
        default=None,
        description="Grouping of teams or players based on gender, age, or skill.",
        examples=["Female"]
    )
    league: Optional[str] = Field(
        default=None,
        description="League of the playing teams.",
        examples=["A"]
    )
    home_team: Optional[str] = Field(
        default=None,
        description="The name of the home team.",
        examples=["FTC", "BVSC"]
    )
    away_team: Optional[str] = Field(
        default=None,
        description="The name of the away team.",
        examples=["BVSC", "FTC"]
    )
    playing_field: Optional[str] = Field(
        default=None,
        description="The name of the playing field where the match will take place.",
        examples=["Szőnyi úti fedett uszoda"]
    )
    scheduled_match_time: Optional[str] = Field(
        default=None,
        description="The sheduled starting time of the match.",
        examples=["2025-06-06 12:30:00"]
    )
    stream_token: str = Field(
        ...,  # 👈 explicitly required
        description="The stream token passed for live streaming",
        examples=["abcd-1234-efgh-5678"]
    )
