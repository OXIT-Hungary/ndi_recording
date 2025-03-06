import os

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

from app.schemas.youtube_stream import YoutubeStreamSchedule


class YoutubeStreamService:
    def __init__(self):
        self.SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
        self.credentials = self._load_credentials()
        self.youtube = build("youtube", "v3", credentials=self.credentials)

    def _load_credentials(self):
        if os.path.exists("../../credentials.json"):
            return Credentials.from_authorized_user_file("../../credentials.json", self.SCOPES)

        oauth_flow = InstalledAppFlow.from_client_config(
            "../../client_secrets.json",  # Download from Google Cloud Console
            self.SCOPES
        )

        credentials = oauth_flow.run_local_server(port=0)

        with open("../../credentials.json", "w") as token:
            token.write(credentials.to_json())

        return credentials

    def schedule_stream(self, stream_details: YoutubeStreamSchedule):
        try:
            broadcast_response = self.youtube.liveBroadcast().insert(
                part="snippet,status,contentDetails",
                body={
                    "snippet": {
                        "title": stream_details.stream_title,
                        "description": stream_details.stream_description,
                        "scheduledStartTime": stream_details.start_time.isoformat() + 'Z',
                        "scheduledEndTime": stream_details.end_time.isoformat() + 'Z'
                    },
                    "status": {
                        "privacyStatus": stream_details.stream_privacy_status,
                        "selfDeclaredMadeForKids": False
                    },
                    "contentDetails": {
                        "enableAutoStart": True,
                        "enableAutoStop": True
                    }
                }
            ).execute()

            stream_response = self.youtube.liveStreams().instert(
                part="snippet,cdn,status",
                body={
                    "snippet": {
                        "title": stream_details.stream_title
                    },
                    "cdn": {
                        "resolution": "variable",
                        "frameRate": "variable"
                    },
                    "status": {
                        "privacyStatus": "private"
                    }
                }
            ).execute()

            self.youtube.liveBroadcast().bind(
                part="id,snippet",
                body={
                    "id": broadcast_response["id"],
                    "snippet": {
                        "resourceId": {
                            "kind": "youtube#liveStream",
                            "id": stream_response["id"]
                        }
                    }
                }
            ).execute()


        except Exception as e:
            raise ValueError(f"Failed to schedule Youtube Stream {str(e)}")

    def cancel_steam(self, broadcast_id: str):
        try:
            self.youtube.liveBroadcast().delete(
                id=broadcast_id
            ).execute()

            return {
                "status": "success",
                "message": f"Broadcast {broadcast_id} has been canceled and deleted"
            }
        except Exception as e:
            raise ValueError(f"Failed to cancel Youtube stream: {e}")

    def get_scheduled_streams(self):
        try:
            response = self.youtube.liveBroadcast().list(
                part="snippet,status",
                broadcastStatus="upcoming",
                maxResults="50"
            ).execute()

            scheduled_streams = []
            for stream in response.get("items", []):
                scheduled_stream = {
                    "id": stream["id"],
                    "title": stream["snippet"]["title"],
                    "scheduled_start_time": stream["snippet"]["scheduledStartTime"],
                    "scheduled_end_time": stream["snippet"]["scheduledEndTime"],
                    "privacy_status": stream["status"]["privacyStatus"]
                }
                scheduled_streams.append(scheduled_stream)

            return scheduled_streams

        except Exception as e:
            raise ValueError(f"Failed to get scheduled streams: {e}")