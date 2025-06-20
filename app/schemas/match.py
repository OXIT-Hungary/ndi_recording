from pydantic import BaseModel, Field
from typing import Optional

class Match(BaseModel):
    id: Optional[int] = Field(
        default=None,
        description="Unique identifier for the match.",
        example=1
    )
    division: Optional[str] = Field(
        default=None,
        description="Grouping of teams or players based on gender, age, or skill.",
        examples=["Female"]
    )
    league: Optional[str] = Field(
        default=None,
        description="Competition or organization the match belongs to.",
        examples=["A"]
    )
    home_team: Optional[str] = Field(
        default=None,
        description="The team hosting the match.",
        examples=["BVSC"]
    )
    away_team: Optional[str] = Field(
        default=None,
        description="The visiting team.",
        examples=["FTC"]
    )
    playing_field: Optional[str] = Field(
        default=None,
        description="The field where the match will be played.",
        examples=["Szőnyi úti fedett uszoda"]
    )
    scheduled_match_time: Optional[str] = Field(
        default=None,
        description="Scheduled date and time of the match in format YYYY-MM-DD HH:MM:SS.",
        examples=["2025-06-06 12:30:00"]
    )
