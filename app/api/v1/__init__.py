from fastapi import APIRouter

from .camera import router as camera_router
from .schedule import router as schedule_router
from .streaming import youtube_router
from .version import router as version_router

router = APIRouter(prefix="/v1")
router.include_router(schedule_router)
router.include_router(camera_router)
router.include_router(version_router)
router.include_router(youtube_router)
