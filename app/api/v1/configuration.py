import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# NOTE: as we are not using the youtube API we don't need this
# load_dotenv()


class Settings(BaseSettings):
    PROJECT_NAME: str = "Youtube Stream Manager"
    PROJECT_DESCRIPTION: str = "This is a Youtube manager where you can schedule streams"

    TOKEN_FILE_PATH: str = os.environ.get("TOKEN_FILE_PATH")

    YOUTUBE_PROJECT_ID: str = os.environ.get("YOUTUBE_PROJECT_ID")
    YOUTUBE_CLIENT_ID: str = os.environ.get("YOUTUBE_CLIENT_ID")
    YOUTUBE_CLIENT_SECRET: str = os.environ.get("YOUTUBE_CLIENT_SECRET")
    YOUTUBE_REDIRECT_URI: str = os.environ.get("YOUTUBE_REDIRECT_URI")
    YOUTUBE_API_VERSION: str = os.environ.get("YOUTUBE_API_VERSION")
    YOUTUBE_SERVICE_NAME: str = os.environ.get("YOUTUBE_SERVICE_NAME")
    YOUTUBE_SERVICE_SCOPE: str = os.environ.get("YOUTUBE_SERVICE_SCOPE")

    GOOGLE_AUTH_URI: str = os.environ.get("GOOGLE_AUTH_URI")
    GOOGLE_TOKEN_URI: str = os.environ.get("GOOGLE_TOKEN_URI")
    GOOGLE_AUTH_CERT_URL: str = os.environ.get("GOOGLE_AUTH_CERT_URL")

    class Config:
        env_file = "../.env"
        case_sensitive = True
        extra = "ignore"

# NOTE: as we are not using the youtube API we don't need this
# settings = Settings()