from fastapi import APIRouter, HTTPException, Depends
from ...schemas.stream_start_request import StreamStartRequest
from src.camera.camera_system import CameraSystem
import datetime
from typing import Optional
from src.config import load_config, Config
from .authenticator import user_or_admin_auth
import logging
from .database_router import Database

logger = logging.getLogger(__name__)

class ManualControlRouter:
    def __init__(self):
        self.camera_system: Optional[CameraSystem] = None
        self.config: Config = load_config(file_path='./configs/bvsc_config.yaml')

    def get_router(self) -> APIRouter:
        router = APIRouter(prefix="/manual_control", tags=["Manual Control"], dependencies=[Depends(user_or_admin_auth)])

        @router.post("/start-stream")
        def start_stream(payload: StreamStartRequest) -> dict:
            try:
                print(f"Starting stream using stream token: {payload.stream_token}")
                self.start_stream(self.config, payload.stream_token)
                Database.save_to_txt(payload)
                Database.insert_match(payload)
                return {"message": f"{datetime.datetime.now()}: Stream started successfully. ", "isRunning": False if self.camera_system is None else self.camera_system.is_running()}
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to START stream: {str(e)}")
            
        @router.post("/stop-stream")
        def stop_stream() -> dict:
            try:
                self.stop_stream()
                return {"message": f"{datetime.datetime.now()}: Stream stopped successfully", "isRunning": False if self.camera_system is None else self.camera_system.is_running()}
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to STOP stream: {str(e)}")
            
        @router.get("/get-stream-status")
        def get_stream_status() -> dict:
            return {"isRunning": False if self.camera_system is None else self.camera_system.is_running()}

        return router

    def start_stream(self, config: Config, stream_token: str) -> None:
        """ Start the camera system and streaming """

        if self.camera_system is not None and self.camera_system.is_running():
            raise RuntimeError(f"Stream is already running.")

        try:
            
            # Initialize camera system only when starting (lazy initialization)
            self.camera_system = CameraSystem(config=config, stream_token=stream_token)
            self.camera_system.start()
        except Exception as e:
            logging.error(f"Failed to start stream: {e}")
            raise RuntimeError(f"Failed to start stream: {e}")

    def stop_stream(self) -> None:
        """ Stop the camera system and streaming. """
        if self.camera_system is None or not self.camera_system.is_running():
            raise RuntimeError("Stream is not running")

        try:
            if self.camera_system:
                if self.camera_system.stop():
                    self.camera_system = None
                
        except Exception as e:

            logging.error(f"Failed to stop stream: {e}")
            raise RuntimeError(f"Failed to stop stream: {e}")
