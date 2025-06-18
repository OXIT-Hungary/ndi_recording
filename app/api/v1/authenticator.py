from fastapi import Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from typing import Optional
from dotenv import load_dotenv
import os


# Load the .env.production file that stores data for the authenticator
env_path = os.path.join(os.path.dirname(__file__), ".env.production")
if load_dotenv(dotenv_path=env_path):
    print(f"[INFO] Loaded environment from: {env_path}")
else:
    print(f"[WARNING] Failed to load environment from: {env_path}")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

"""

Example client request: 

POST /start-stream
Headers:
  X-API-Key: super-secret-api-key-123456789
Body:
  {
    "stream_token": "abcd-1234-efgh-5678"
  }

"""

def validate_api_key(api_key: Optional[str] = Depends(api_key_header)) -> str:
    if api_key == os.getenv("API_KEY"):
        return api_key
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid or missing API Key"
    )