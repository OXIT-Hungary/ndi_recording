import datetime
import threading

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.api.services.youtube_service import youtube_service
from app.schemas.youtube_stream import YoutubeStreamSchedule
from src.config import load_config
from src.camera import camera_system as camera_sys

templates = Jinja2Templates(directory="app/templates/streaming")

youtube_router = APIRouter(prefix="/youtube", tags=["youtube"])

cfg = load_config(file_path='./configs/default_config.yaml')
cfg.court_width = 25
cfg.court_height = 20
camera_system = None
stream_timer = None


class StreamTimer:
    def __init__(self, camera_sys, start_time, end_time):
        self.camera_sys = camera_sys
        self.start_time = start_time
        self.end_time = end_time
        self.start_timer = None
        self.end_timer = None
        self.running = False

    def schedule_stream(self):
        """Schedule the start and end of the stream at the specified times"""
        # Calculate seconds until start and end
        now = datetime.datetime.now(datetime.timezone.utc)
        seconds_until_start = max(0, (self.start_time - now).total_seconds())
        seconds_until_end = max(0, (self.end_time - now).total_seconds())

        print(f"Stream will start in {seconds_until_start:.1f} seconds")
        print(f"Stream will end in {seconds_until_end:.1f} seconds")

        # Schedule start timer if start time is in the future
        if seconds_until_start > 0:
            self.start_timer = threading.Timer(seconds_until_start, self.start_stream)
            self.start_timer.daemon = True
            self.start_timer.start()
        else:
            # Start immediately if start time is in the past
            self.start_stream()

        # Schedule end timer
        if seconds_until_end > 0:
            self.end_timer = threading.Timer(seconds_until_end, self.end_stream)
            self.end_timer.daemon = True
            self.end_timer.start()

    def start_stream(self):
        """Start the camera system and streaming"""
        print(f"Starting stream at {datetime.datetime.now()}")
        if self.camera_sys and not self.running:
            self.camera_sys.start()
            self.running = True

    def end_stream(self):
        """Stop the camera system and streaming"""
        print(f"Ending stream at {datetime.datetime.now()}")
        if self.camera_sys and self.running:
            self.camera_sys.stop()
            self.running = False

    def cancel(self):
        """Cancel all scheduled timers"""
        if self.start_timer:
            self.start_timer.cancel()
        if self.end_timer:
            self.end_timer.cancel()
        if self.running:
            self.end_stream()


@youtube_router.get("/auth")
def start_auth_process():
    try:
        auth_url = youtube_service.get_auth_url()
        return RedirectResponse(url=auth_url)
    except Exception as e:
        error_message = f"Error starting authentication: {e}"
        return RedirectResponse(url=f"/?error={error_message}")


@youtube_router.get("/auth/callback", response_class=HTMLResponse)
def auth_callback(request: Request, code: str = Query(None), error: str = Query(None)):
    try:
        if error:
            error_message = f"OAuth error: {error}"
            return templates.TemplateResponse("error.html", {"request": request, "error_message": error_message})

        if not code:
            error_message = "No authorization code provided in callback"
            return templates.TemplateResponse("error.html", {"request": request, "error_message": error_message})

        youtube_service.handle_oauth_callback(code)

        if youtube_service.is_authenticated():
            return templates.TemplateResponse("success.html", {"request": request})
        else:
            error_message = "Authentication failed. Please try again."
            return templates.TemplateResponse("error.html", {"request": request, "error_message": error_message})
    except Exception as e:
        error_message = f"Error during authentication: {str(e)}"
        return templates.TemplateResponse("error.html", {"request": request, "error_message": error_message})


@youtube_router.get("/logout")
def logout():
    try:
        global stream_timer
        if stream_timer:
            stream_timer.cancel()
            stream_timer = None

        youtube_service.clear_credentials()
        return RedirectResponse(url="/?message=Successfully logged out")
    except Exception as e:
        error_message = f"Error during logging out: {e}"
        return RedirectResponse(url=f"/?error={error_message}")


@youtube_router.get("/streams", response_class=HTMLResponse)
def get_scheduled_streams(request: Request):
    try:
        if not youtube_service.is_authenticated():
            return RedirectResponse(url="/?error=Not authenticated. Please authenticate first.")

        scheduled_streams = youtube_service.list_streams()
        return templates.TemplateResponse("streams.html", {"request": request, "streams": scheduled_streams})
    except Exception as e:
        error_message = f"Error fetching streams: {str(e)}"
        return templates.TemplateResponse("error.html", {"request": request, "error_message": error_message})


@youtube_router.post("/create-scheduled-streams")
def create_scheduled_streams(stream_details: YoutubeStreamSchedule, request: Request):
    try:
        if not youtube_service.is_authenticated():
            return RedirectResponse(url="/?error=Not authenticated. Please authenticate first.")

        # Create the YouTube stream
        new_stream = youtube_service.create_scheduled_stream(stream_details)

        # Initialize camera system but don't start it yet
        global camera_system, stream_timer

        # Cancel any existing stream timer
        if stream_timer:
            stream_timer.cancel()

        # Initialize camera system without starting it
        camera_system = camera_sys.CameraSystem(
            config=cfg.camera_system, out_path=cfg.out_path, stream_token=new_stream['stream_key']
        )

        # Create and start the stream timer to handle automatic start/stop
        stream_timer = StreamTimer(
            camera_system,
            stream_details.start_time,
            stream_details.end_time
        )
        stream_timer.schedule_stream()

        # Return success to the frontend
        return {"status": "success", "message": "Stream scheduled successfully"}

    except Exception as e:
        error_message = f"Error creating stream: {str(e)}"
        return {"status": "error", "detail": error_message}


@youtube_router.delete("/delete/{stream_id}")
def delete_stream(stream_id: str, request: Request):
    try:
        if not youtube_service.is_authenticated():
            return {"status": "error", "message": "Not authenticated. Please authenticate first."}

        # Stop the stream timer if we're deleting the active stream
        global stream_timer
        if stream_timer:
            stream_timer.cancel()
            stream_timer = None

        success = youtube_service.delete_stream(stream_id)
        if success:
            return {"status": "success", "message": f"Stream {stream_id} deleted successfully"}
        return {"status": "error", "message": f"Failed to delete stream {stream_id}"}
    except Exception as e:
        error_message = f"Error deleting stream: {str(e)}"
        return {"status": "error", "message": error_message}