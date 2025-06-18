from fastapi import APIRouter

# from .schedule import router as schedule_router
# from .streaming import youtube_router
from .version import router as version_router
from .manual_control import ManualControlRouter

from shared_manager import SharedManager
SharedManager.init()

router = APIRouter(prefix="/v1")
# router.include_router(schedule_router)
router.include_router(version_router)
# router.include_router(youtube_router)

manual_control_router = ManualControlRouter()
router.include_router(manual_control_router.get_router())