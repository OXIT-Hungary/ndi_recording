from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .api import router as api_router

# NOTE: as we are not using the youtube API we don't need this
# from .api.services.youtube_service import youtube_service

from .core.utils.custom_unique_id import custom_generate_unique_id

templates_dir = Path("app/templates/streaming")
if not templates_dir.exists():
    templates_dir.mkdir()
templates = Jinja2Templates(directory=templates_dir)


app = FastAPI(
    generate_unique_id_function=custom_generate_unique_id,
    title="OXCAM",
    version="v0.3.0",
    contact={"name": "OXIT", "email": "contact@oxit.hu"},
)
app.include_router(api_router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root(request: Request, error: str = None, message: str = None):
    # NOTE: as we are not using the youtube API we don't need this
    # is_authenticated = youtube_service.is_authenticated()

    return templates.TemplateResponse(
        "redirect.html", {"request": request}
    )
