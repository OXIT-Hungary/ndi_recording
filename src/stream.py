import subprocess


class YouTubeStream:
    def __init__(self, token: str):
        self.token = token

        self.ffmpeg = None

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass
