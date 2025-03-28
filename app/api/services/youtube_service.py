import os
import pickle
from datetime import datetime, timezone, timedelta

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from dotenv import load_dotenv

from app.schemas.youtube_stream import YoutubeStreamSchedule
from app.api.v1.configuration import settings


class YoutubeService:
    def __init__(self):
        load_dotenv()
        self.credentials = None
        self.youtube = None

    def _load_credentials(self):
        if os.path.exists(settings.TOKEN_FILE_PATH):
            try:
                with open(settings.TOKEN_FILE_PATH, 'rb') as token:
                    credentials = pickle.load(token)

                if credentials and credentials.valid:
                    # Load valid credentials from token file
                    self.credentials = credentials
                    self.youtube = build(settings.YOUTUBE_SERVICE_NAME, settings.YOUTUBE_API_VERSION, credentials=credentials)
                    print("Credentials found and loaded successfully")
                    return

                elif credentials and credentials.expired and credentials.refresh_token:
                    try:
                        # Refresh expired credentials
                        credentials.refresh(Request())
                        with open(settings.TOKEN_FILE_PATH, 'wb') as token:
                            pickle.dump(credentials, token)
                        self.credentials = credentials
                        self.youtube = build(settings.YOUTUBE_SERVICE_NAME, settings.YOUTUBE_API_VERSION, credentials=credentials)
                        print("Credentials found and refreshed the token successfully")
                        return
                    except Exception as e:
                        print(f"Error refreshing credentials: {e}")
            except Exception as e:
                print(f"Error loading credentials from token file: {e}")
        print("No valid credentials found")

    def _create_oauth_flow(self):
        try:
            if not all([settings.YOUTUBE_CLIENT_ID, settings.YOUTUBE_CLIENT_SECRET, settings.YOUTUBE_REDIRECT_URI]):
                print("Youtube API credentials are missing")
                raise ValueError("Youtube API credentials are missing")

            client_config = self._get_client_secret()

            flow = Flow.from_client_config(
                client_config,
                scopes=[settings.YOUTUBE_SERVICE_SCOPE],
                redirect_uri=settings.YOUTUBE_REDIRECT_URI
            )
            return flow
        except Exception as e:
            print(f"The following error occurred during the oauth flow creation: {e}")

    def get_auth_url(self):
        flow = self._create_oauth_flow()
        auth_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent"
        )
        return auth_url

    def handle_oauth_callback(self, code: str):
        try:
            flow = self._create_oauth_flow()
            flow.fetch_token(code=code)
            credentials = flow.credentials

            # save credentials
            with open(settings.TOKEN_FILE_PATH, "wb") as file:
                pickle.dump(credentials, file)
            self.credentials = credentials
            self.youtube = build(settings.YOUTUBE_SERVICE_NAME, settings.YOUTUBE_API_VERSION, credentials=credentials)
        except Exception as e:
            print(f"Error handling OAuth callback: {e}")

    def is_authenticated(self):
        if self.credentials is not None and self.youtube is not None:
            return True
        return False

    def list_streams(self):
        if not self.is_authenticated():
            print("Not authenticated user")
            raise ValueError("Not authenticated with Youtube API")

        try:
            # Call the YouTube API to get upcoming broadcasts
            broadcasts = self.youtube.liveBroadcasts().list(
                part="id,snippet,status",
                broadcastType="all",
                mine=True
            ).execute()

            current_time = datetime.now(timezone.utc)
            active_streams = []

            for broadcast in broadcasts.get("items", []):
                # Check lifecycle status for active streams
                lifecycle_status = broadcast["status"]["lifeCycleStatus"]
                active_statuses = ["live", "ready", "testStarting"]

                # Parse scheduled times, converting to datetime with UTC timezone
                scheduled_start = broadcast["snippet"].get("scheduledStartTime")
                scheduled_end = broadcast["snippet"].get("scheduledEndTime")

                # Convert scheduled times to datetime objects if they exist
                start_time = datetime.fromisoformat(scheduled_start.replace('Z', '+00:00')) if scheduled_start else None
                end_time = datetime.fromisoformat(scheduled_end.replace('Z', '+00:00')) if scheduled_end else None

                # Determine if stream is active based on status or time
                is_time_active = (
                        start_time and end_time and
                        start_time <= current_time <= end_time
                )

                is_status_active = lifecycle_status in active_statuses

                if is_status_active or is_time_active:
                    # Format times: add 1 hour and remove Z
                    formatted_start = (start_time + timedelta(hours=1)).strftime(
                        "%Y-%m-%d %H:%M") if start_time else None
                    formatted_end = (end_time + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M") if end_time else None

                    stream = {
                        "stream_id": broadcast["id"],
                        "title": broadcast["snippet"]["title"],
                        "description": broadcast["snippet"]["description"],
                        "scheduled_start_time": formatted_start,
                        "scheduled_end_time": formatted_end,
                        "status": lifecycle_status
                    }
                    active_streams.append(stream)

            return active_streams
        except Exception as e:
            print(f"Error listing streams: {str(e)}")
            raise ValueError(f"Error listing streams: {str(e)}")

    def create_live_stream(self, stream_details: YoutubeStreamSchedule):
        try:
            request = self.youtube.liveStreams().insert(
                part="snippet,cdn,contentDetails,status",
                body={
                    "snippet": {
                        "title": stream_details.title,
                        "description": stream_details.description
                    },
                    "cdn": {
                        "resolution": "variable",
                        "frameRate": "variable",
                        "ingestionType": "rtmp"
                    },
                    "contentDetails": {
                        "enableAutoStart": True,
                        "isReusable": True
                    }
                }
            )

            response = request.execute()

            return response["id"]
        except Exception as e:
            print(f"Error creating live stream: {e}")
            raise

    def create_live_broadcast(self,stream_details: YoutubeStreamSchedule, stream_id: str):
        request = self.youtube.liveBroadcasts().insert(
            part="snippet,status,contentDetails",
            body={
                "snippet": {
                    "title": stream_details.title,
                    "description": stream_details.description,
                    "scheduledStartTime": stream_details.start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "scheduledEndTime": stream_details.end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
                },
                "status": {
                    "privacyStatus": stream_details.privacy_status
                },
                "contentDetails": {
                    "enableAutoStart": True
                }
            }
        )
        response = request.execute()
        broadcast_id = response["id"]

        # Bind the stream to the live broadcast
        bind_request = self.youtube.liveBroadcasts().bind(
            part="id,snippet",
            id=broadcast_id,
            streamId=stream_id
        )
        bind_response = bind_request.execute()

        return broadcast_id

    def start_live_broadcast(self, broadcast_id: str):
        request = self.youtube.liveBroadcasts().update(
            part="id,status",
            body={
                "id": broadcast_id,
                "status": {
                    "lifeCycleStatus": "live"
                }
            }
        )
        response = request.execute()
        # print(f"TOKEN: {request['http']['credentials']['token']}")
        return response["id"]

    def get_stream_details(self, broadcast_id: str):
        if not self.is_authenticated():
            raise ValueError("Not authenticated with YouTube API")

        try:
            # Retrieve the live broadcast details
            broadcast_response = self.youtube.liveBroadcasts().list(
                part="snippet,contentDetails",
                id=broadcast_id
            ).execute()

            if not broadcast_response.get('items'):
                raise ValueError(f"No broadcast found with ID {broadcast_id}")

            # Get the associated stream
            stream_id = broadcast_response['items'][0]['contentDetails']['boundStreamId']

            # Retrieve the stream details
            stream_response = self.youtube.liveStreams().list(
                part="cdn",
                id=stream_id
            ).execute()

            if not stream_response.get('items'):
                raise ValueError(f"No stream found with ID {stream_id}")

            # Extract stream key and ingestion address
            stream_details = stream_response['items'][0]['cdn']['ingestionInfo']

            return {
                "stream_key": stream_details['streamName'],
                "ingestion_address": stream_details['ingestionAddress'],
                "broadcast_id": broadcast_id,
                "stream_id": stream_id
            }

        except Exception as e:
            print(f"Error retrieving stream details: {e}")
            raise ValueError(f"Error retrieving stream details: {str(e)}")

    def create_scheduled_stream(self, stream_details: YoutubeStreamSchedule):
        if not self.is_authenticated():
            raise ValueError("Not authenticated with YouTube API")

        try:
            stream_id = self.create_live_stream(stream_details)
            broadcast_id = self.create_live_broadcast(stream_details, stream_id)
            self.start_live_broadcast(broadcast_id)

            # Retrieve and return stream details
            stream_info = self.get_stream_details(broadcast_id)
            print("Successfully started stream")
            return stream_info

        except Exception as e:
            print(f"Error scheduling stream: {e}")
            raise

    def delete_stream(self, stream_id: str) -> bool:
        if not self.is_authenticated():
            raise ValueError("Not authenticated with YouTube API")

        try:
            self.youtube.liveBroadcasts().delete(
                id=stream_id
            ).execute()
            return True
        except Exception as e:
            print(f"Error deleting stream: {e}")
            return False


    @staticmethod
    def _get_client_secret():
        client_config = {
            "web": {
                "client_id": settings.YOUTUBE_CLIENT_ID,
                "project_id": settings.YOUTUBE_PROJECT_ID,
                "auth_uri": settings.GOOGLE_AUTH_URI,
                "token_uri": settings.GOOGLE_TOKEN_URI,
                "auth_provider_x509_cert_url": settings.GOOGLE_AUTH_CERT_URL,
                "client_secret": settings.YOUTUBE_CLIENT_SECRET,
                "redirect_uris": [settings.YOUTUBE_REDIRECT_URI],
                "javascript_origins": ["http://localhost:8000"]
            }
        }
        return client_config


youtube_service = YoutubeService()
