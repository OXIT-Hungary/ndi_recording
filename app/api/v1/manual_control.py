from fastapi import APIRouter, HTTPException, Depends
from ...schemas.manual_control import StreamStartRequest
from src.camera.camera_system import CameraSystem
import datetime
from typing import Optional
from src.config import load_config, Config
from .authenticator import validate_api_key
from shared_manager import SharedManager, StreamStatus


class ManualControlRouter:
    def __init__(self):
        self.camera_system: Optional[CameraSystem] = None
        self.config: Config = load_config(file_path='./configs/bvsc_config.yaml')

        SharedManager.stream_status.value = StreamStatus.UNDEFINED

    def get_router(self) -> APIRouter:
        router = APIRouter(prefix="/manual_control", tags=["Manual Control"], dependencies=[Depends(validate_api_key)])

        @router.post("/start-stream")
        def start_stream(payload: StreamStartRequest) -> dict:
            try:
                self.start_stream(self.config, payload.stream_token)
                return {"message": f"{datetime.datetime.now()}: Stream started successfully at ", "status": StreamStatus(SharedManager.stream_status.value).name}
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to START stream: {str(e)}")
            
        @router.post("/stop-stream")
        def stop_stream() -> dict:
            try:
                self.stop_stream()
                return {"message": f"{datetime.datetime.now()}: Stream stopped successfully", "status": StreamStatus(SharedManager.stream_status.value).name}
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to STOP stream: {str(e)}")
            
        @router.get("/get-stream-status")
        def get_stream_status() -> dict:
            return {"status": StreamStatus(SharedManager.stream_status.value).name}

        return router

    def start_stream(self, config: Config, stream_token: str) -> None:
        """ Start the camera system and streaming """

        if SharedManager.stream_status.value == StreamStatus.STARTING or SharedManager.stream_status.value == StreamStatus.STREAMING:
            raise RuntimeError("Stream already started")

        try:
            SharedManager.stream_status.value = StreamStatus.STARTING
            # Initialize camera system only when starting (lazy initialization)
            self.camera_system = CameraSystem(config=config, stream_token=stream_token)
            self.camera_system.start()
        except Exception as e:
            SharedManager.stream_status.value = StreamStatus.ERROR_STARTING

            print(f"[ERROR] Failed to start stream: {e}")
            raise RuntimeError(f"Error starting stream: {e}")

    def stop_stream(self) -> None:
        """ Stop the camera system and streaming. """
        
        if SharedManager.stream_status.value == StreamStatus.STOPPED:
            raise RuntimeError("Stream is not running")

        try:
            if self.camera_system:
                if self.camera_system.stop():
                    self.camera_system = None
                
                SharedManager.stream_status.value = StreamStatus.STOPPED

        except Exception as e:
            SharedManager.stream_status.value = StreamStatus.ERROR_STOPPING

            print(f"[ERROR] Failed to stop stream: {e}")
            raise RuntimeError(f"Error ending stream: {e}")
