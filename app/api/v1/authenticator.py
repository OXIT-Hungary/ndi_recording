import os
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from typing import Optional
from dotenv import load_dotenv
import logging


class Authenticator:
    def __init__(self, allow_user_key: bool = False):
        self.allow_user_key = allow_user_key

        self.env_path = os.path.join(os.path.abspath(__file__), ".env.production")
        self.load_env()

    def load_env(self):
        load_dotenv(dotenv_path=self.env_path)

    async def __call__(self, api_key: Optional[str] = Depends(APIKeyHeader(name="X-API-Key", auto_error=False))):
        admin_key = os.getenv("ADMIN_API_KEY")
        user_key = os.getenv("USER_API_KEY")

        if api_key == admin_key:
            return api_key

        if self.allow_user_key and api_key == user_key:
            return api_key

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Key"
        )

# Dependency instances
admin_only_auth = Authenticator(allow_user_key=False)
user_or_admin_auth = Authenticator(allow_user_key=True)