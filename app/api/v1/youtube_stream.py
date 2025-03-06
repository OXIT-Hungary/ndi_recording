from fastapi import APIRouter, Depends, HTTPException

from app.core.youtube_service import YoutubeStreamService
from app.schemas.youtube_stream import YoutubeStreamSchedule

router = APIRouter(prefix="/youtube", tags=["Youtube Streaming"])


@router.get("/get-scheduled-streams")
def get_scheduled_youtube_streams(youtube_service: YoutubeStreamService = Depends()):
    try:
        scheduled_streams = youtube_service.get_scheduled_streams()
        return scheduled_streams
    except Exception as e:
        raise HTTPException(detail=f"Failed to get the scheduled streams: {str(e)}", status_code=400)


@router.post("/schedule-stream")
def schedule_youtube_stream(stream_details: YoutubeStreamSchedule, youtube_service: YoutubeStreamService = Depends()):
    try:
        stream_info = youtube_service.schedule_stream(stream_details)

        return {
            "message": "Stream scheduled successfully",
            "stream_details": stream_info
        }
    except Exception as e:
        raise HTTPException(detail=f"Failed to schedule a stream: {str(e)}", status_code=400)


@router.delete("/cancel-scheduled-stream/{broadcast_id}")
def cancel_scheduled_youtube_stream(broadcast_id: str, youtube_service: YoutubeStreamService = Depends()):
    try:
        cancellation_result = youtube_service.cancel_steam(broadcast_id)
        return cancellation_result
    except Exception as e:
        raise HTTPException(detail=f"Failed to cancel stream: {str(e)}", status_code=400)