from fastapi import APIRouter, HTTPException, Depends
from ...schemas.manual_control import StreamStartRequest
from src.camera.camera_system import CameraSystem
import datetime
from typing import Optional
from src.config import load_config, Config
from .authenticator import validate_api_key
import shared_manager as shared_manager


class ManualControlRouter:
    def __init__(self):
        self.camera_system: Optional[CameraSystem] = None
        self.config: Config = load_config(file_path='./configs/fradi_config.yaml')

        shared_manager.stream_status.value = shared_manager.StreamStatus.UNDEFINED

    def get_router(self) -> APIRouter:
        router = APIRouter(prefix="/manual_control", tags=["Manual Control"], dependencies=[Depends(validate_api_key)])

        @router.post("/start-stream")
        def start_stream(payload: StreamStartRequest) -> dict:
            try:
                self.start_stream(self.config, payload.stream_token)
                return {"message": f"{datetime.datetime.now()}: Stream started successfully at ", "status": self.stream_status}
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to START stream: {str(e)}")
            
        @router.post("/stop-stream")
        def stop_stream() -> dict:
            try:
                self.end_stream()
                return {"message": f"{datetime.datetime.now()}: Stream stopped successfully", "status": self.stream_status}
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to STOP stream: {str(e)}")
            
        @router.get("/get-stream-status")
        def get_stream_status() -> dict:
            return {"status": shared_manager.stream_status.value}

        return router

    def start_stream(self, config: Config, stream_token: str) -> None:
        """ Start the camera system and streaming """

        if shared_manager.stream_status.value == shared_manager.StreamStatus.STARTING or shared_manager.stream_status.value == shared_manager.StreamStatus.STREAMING:
            raise RuntimeError("Stream already started")

        try:
            shared_manager.stream_status.value = shared_manager.StreamStatus.STARTING
            # Initialize camera system only when starting (lazy initialization)
            self.camera_system = CameraSystem(config=config, stream_token=stream_token)
            self.camera_system.start()
        except Exception as e:
            shared_manager.stream_status.value = shared_manager.StreamStatus.ERROR

            print(f"[ERROR] Failed to start stream: {e}")
            raise RuntimeError(f"Error starting stream: {e}")


    def end_stream(self) -> None:
        """ Stop the camera system and streaming. """
        
        if shared_manager.stream_status.value == shared_manager.StreamStatus.STOPPED:
            raise RuntimeError("Stream is not running")

        try:
            if self.camera_system:
                if self.camera_system.stop():
                    self.camera_system = None
                
                shared_manager.stream_status.value = shared_manager.StreamStatus.STOPPED

        except Exception as e:
            shared_manager.stream_status.value = shared_manager.StreamStatus.ERROR

            print(f"[ERROR] Failed to stop stream: {e}")
            raise RuntimeError(f"Error ending stream: {e}")
