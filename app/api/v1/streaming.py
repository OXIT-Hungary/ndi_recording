import datetime

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.api.services.youtube_service import youtube_service
from app.schemas.youtube_stream import YoutubeStreamSchedule
from src.config import load_config
from src.camera import camera_system as camera_sys
from src.stream import YouTubeStream


templates = Jinja2Templates(directory="app/templates/streaming")

youtube_router = APIRouter(prefix="/youtube", tags=["youtube"])

cfg = load_config(file_path='./configs/default_config.yaml')
cfg.court_width = 25
cfg.court_height = 20
camera_system = camera_sys.CameraSystem(config=cfg, out_path=cfg.out_path)


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

        new_stream = youtube_service.create_scheduled_stream(stream_details)
        yt_stream = YouTubeStream(token=new_stream['stream_key'])

        camera_system.set_stream(yt_stream)
        camera_system.start()

    except Exception as e:
        error_message = f"Error creating stream: {str(e)}"
        return templates.TemplateResponse("error.html", {"request": request, "error_message": error_message})


@youtube_router.delete("/delete/{stream_id}")
def delete_stream(stream_id: str, request: Request):
    try:
        if not youtube_service.is_authenticated():
            return RedirectResponse(url="/?error=Not authenticated. Please authenticate first.")

        success = youtube_service.delete_stream(stream_id)
        if success:
            return {"status": "success", "message": f"Stream {stream_id} deleted successfully"}
        return {"status": "error", "message": f"Failed to delete stream {stream_id}"}
    except Exception as e:
        error_message = f"Error deleting stream: {str(e)}"
        return templates.TemplateResponse("error.html", {"request": request, "error_message": error_message})
