from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .api import router as api_router
from .api.services.youtube_service import youtube_service
from .core.utils.custom_unique_id import custom_generate_unique_id



templates_dir = Path("app/templates/streaming")
if not templates_dir.exists():
    templates_dir.mkdir()
templates = Jinja2Templates(directory=templates_dir)



app = FastAPI(
    generate_unique_id_function=custom_generate_unique_id,
    title="OXCAM",
    version="v0.2.0",
    contact={"name": "OX-IT", "email": "viktor.koch@oxit.hu"},
)
app.include_router(api_router)


@app.get("/", response_class=HTMLResponse)
def root(request: Request, error: str = None):
    # Check authentication status
    is_authenticated = youtube_service.is_authenticated()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "is_authenticated": is_authenticated,
            "error_message": error
        }
    )
