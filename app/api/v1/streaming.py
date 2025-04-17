import datetime
import threading
from typing import Dict, Optional

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

# Dictionary to store active stream timers by stream ID
stream_timers: Dict[str, 'StreamTimer'] = {}
# Dictionary to track stream statuses
stream_statuses: Dict[str, str] = {}


class StreamTimer:
    def __init__(self, camera_sys_config, stream_token, stream_id, start_time, end_time):
        self.camera_sys = None
        self.camera_sys_config = camera_sys_config
        self.stream_token = stream_token
        self.stream_id = stream_id
        self.start_time = start_time
        self.end_time = end_time
        self.start_timer: Optional[threading.Timer] = None
        self.end_timer: Optional[threading.Timer] = None
        self.running = False

    def schedule_stream(self):
        """Schedule the start and end of the stream at the specified times"""
        # Calculate seconds until start and end
        now = datetime.datetime.now(datetime.timezone.utc)
        seconds_until_start = max(0, (self.start_time - now).total_seconds())
        seconds_until_end = max(0, (self.end_time - now).total_seconds())

        print(f"Stream {self.stream_id} will start in {seconds_until_start:.1f} seconds")
        print(f"Stream {self.stream_id} will end in {seconds_until_end:.1f} seconds")

        # Update global status
        if now >= self.end_time:
            stream_statuses[self.stream_id] = "ended"
        elif now >= self.start_time:
            stream_statuses[self.stream_id] = "live"
        else:
            stream_statuses[self.stream_id] = "scheduled"

        # Schedule start timer if start time is in the future
        if seconds_until_start > 0:
            self.start_timer = threading.Timer(seconds_until_start, self.start_stream)
            self.start_timer.daemon = True
            self.start_timer.start()
        else:
            # Start immediately if start time is in the past and before end time
            if now < self.end_time:
                self.start_stream()
            else:
                # Stream's scheduled time has already passed
                stream_statuses[self.stream_id] = "ended"
                return

        # Schedule end timer
        if seconds_until_end > 0:
            self.end_timer = threading.Timer(seconds_until_end, self.end_stream)
            self.end_timer.daemon = True
            self.end_timer.start()

    def start_stream(self):
        """Start the camera system and streaming"""
        print(f"Starting stream {self.stream_id} at {datetime.datetime.now()}")
        try:
            if not self.running:
                # Initialize camera system only when starting (lazy initialization)
                self.camera_sys = camera_sys.CameraSystem(
                    config=self.camera_sys_config, 
                    out_path=cfg.out_path, 
                    stream_token=self.stream_token
                )
                self.camera_sys.start()
                self.running = True
                # Update status
                stream_statuses[self.stream_id] = "live"
        except Exception as e:
            print(f"Error starting stream {self.stream_id}: {str(e)}")
            stream_statuses[self.stream_id] = "error"

    def end_stream(self):
        """Stop the camera system and streaming"""
        print(f"Ending stream {self.stream_id} at {datetime.datetime.now()}")
        try:
            if self.running and self.camera_sys:
                self.camera_sys.stop()
                self.camera_sys = None  # Release the camera system
                self.running = False
            
            # Update status regardless of whether it was running
            stream_statuses[self.stream_id] = "ended"
            
            # Keep the timer in the dictionary, but mark it as ended
            # We don't delete it so we can show the ended status
        except Exception as e:
            print(f"Error ending stream {self.stream_id}: {str(e)}")
            stream_statuses[self.stream_id] = "error"

    def cancel(self):
        """Cancel all scheduled timers"""
        if self.start_timer:
            self.start_timer.cancel()
            self.start_timer = None
        if self.end_timer:
            self.end_timer.cancel()
            self.end_timer = None
        if self.running and self.camera_sys:
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
        # Cancel all active stream timers
        for stream_id, timer in list(stream_timers.items()):
            timer.cancel()
        stream_timers.clear()
        stream_statuses.clear()

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
        
        # Check for any existing streams that don't have timers and create them
        for stream in scheduled_streams:
            stream_id = stream['stream_id']
            
            # Apply our custom status if we have one
            if stream_id in stream_statuses:
                stream['status'] = stream_statuses[stream_id]
            
            # If stream doesn't have a timer yet and isn't marked as ended, set one up
            if stream_id not in stream_timers and stream.get('status') != "ended":
                # Try to get stream details to set up timer
                try:
                    stream_details = youtube_service.get_stream_details(stream_id)
                    
                    # Parse times from string format to datetime objects with UTC timezone
                    try:
                        start_time = datetime.datetime.strptime(
                            stream['scheduled_start_time'], "%Y-%m-%d %H:%M"
                        )
                        # Adjust timezone - assuming the times are in local time and need to be converted to UTC
                        start_time = start_time.replace(tzinfo=datetime.timezone.utc) - datetime.timedelta(hours=2)
                        
                        end_time = datetime.datetime.strptime(
                            stream['scheduled_end_time'], "%Y-%m-%d %H:%M"
                        )
                        # Adjust timezone - assuming the times are in local time and need to be converted to UTC
                        end_time = end_time.replace(tzinfo=datetime.timezone.utc) - datetime.timedelta(hours=2)
                        
                        # Create and schedule the timer
                        stream_timer = StreamTimer(
                            camera_sys_config=cfg.camera_system,
                            stream_token=stream_details['stream_key'],
                            stream_id=stream_id,
                            start_time=start_time,
                            end_time=end_time
                        )
                        stream_timers[stream_id] = stream_timer
                        stream_timer.schedule_stream()
                        
                        print(f"Set up timer for existing stream {stream_id}")
                    except ValueError as e:
                        print(f"Error parsing date for stream {stream_id}: {str(e)}")
                except Exception as e:
                    print(f"Error setting up timer for existing stream {stream_id}: {str(e)}")
                    # Continue to next stream if there's an error with this one
                    continue
        
        # Add statuses from our tracking system
        for stream in scheduled_streams:
            if stream['stream_id'] in stream_statuses:
                stream['status'] = stream_statuses[stream['stream_id']]
        
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
        stream_id = new_stream['broadcast_id']
        
        # Cancel any existing timer for this stream (just in case)
        if stream_id in stream_timers:
            stream_timers[stream_id].cancel()
        
        # Create a new stream timer for this specific stream
        stream_timer = StreamTimer(
            camera_sys_config=cfg.camera_system,
            stream_token=new_stream['stream_key'],
            stream_id=stream_id,
            start_time=stream_details.start_time,
            end_time=stream_details.end_time
        )
        
        # Store the timer in our dictionary
        stream_timers[stream_id] = stream_timer
        
        # Initially mark as scheduled
        stream_statuses[stream_id] = "scheduled"
        
        # Schedule the stream
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

        # Cancel the specific stream timer if it exists
        if stream_id in stream_timers:
            stream_timers[stream_id].cancel()
            del stream_timers[stream_id]
        
        # Remove from status tracking
        if stream_id in stream_statuses:
            del stream_statuses[stream_id]

        success = youtube_service.delete_stream(stream_id)
        if success:
            return {"status": "success", "message": f"Stream {stream_id} deleted successfully"}
        return {"status": "error", "message": f"Failed to delete stream {stream_id}"}
    except Exception as e:
        error_message = f"Error deleting stream: {str(e)}"
        return {"status": "error", "message": error_message}