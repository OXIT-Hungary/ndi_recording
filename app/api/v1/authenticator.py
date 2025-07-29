import os
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from typing import Optional
from dotenv import load_dotenv
import logging

# Example post for start-stream with authentication
"""
  'http://localhost:8000/api/v1/manual_control/start-stream' \
  -H 'accept: application/json' \
  -H 'X-API-Key: w3bg-nKyfiU9rYskogpLY8AA7qcL8l-8pNVWqoxFF7U' \
  -H 'Content-Type: application/json' \
  -d '{
        "division": "Female",
        "league": "A",
        "home_team": "FTC",
        "away_team": "BVSC",
        "playing_field": "Szőnyi úti fedett uszoda",
        "cheduled_match_time": "2025-06-06 12:30:00",
        "stream_token": "abcd-1234-efgh-5678"
    }'
"""

class Authenticator:
    _env_loaded = False  # Class variable to track if env is loaded

    def __init__(self, allow_user_key: bool = False):
        self.allow_user_key = allow_user_key

        self._ensure_env_loaded()

    @classmethod
    def _ensure_env_loaded(cls):
        # Only load if not already loaded
        if not cls._env_loaded:
            env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.production")
            #print(f"Looking for .env file at: {env_path}")
            
            if os.path.exists(env_path):
                load_dotenv(dotenv_path=env_path)
                print("Environment file loaded successfully!")
            else:
                print("Environment file not found!")
                # Maybe add fallback to load just an .env file?
            
            cls._env_loaded = True
        else:
            print("Environment already loaded, skipping...")

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