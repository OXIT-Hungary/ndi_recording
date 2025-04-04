import subprocess


class YouTubeStream:
    def __init__(self, token: str):
        self.token = token
        self.input_url = "rtmp://192.168.33.101:1935/live/av0"
        self.ffmpeg = None

    def start(self) -> None:
        # fmt: off
        """ self.ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", "1920x1080",
                "-r", "30",
                "-i", self.,
                "-f", "lavfi",
                "-i", "anullsrc",  # Silent audio
                "-c:v", "h264_nvenc",
                "-preset", "ultrafast",
                "-b:v", "3000k",
                "-bufsize", "6000k",
                "-pix_fmt", "yuv420p",
                "-g", "60",
                "-keyint_min", "30",
                "-c:a", "aac",
                "-b:a", "128k",
                "-ar", "44100",
                "-f", "flv",
                f"rtmp://a.rtmp.youtube.com/live2/{self.token}"
            ],
            stdin=subprocess.PIPE,
        ) """
        self.ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-i", self.input_url,
                "-c:v", "copy",
                "-c:a", "aac",
                "-ar", "44100",
                "-b:a", "128k",
                "-f", "flv",
                f"rtmp://a.rtmp.youtube.com/live2/{self.token}"
            ],
            stdout=subprocess.PIPE,
            )
        # fmt: on

    def stop(self) -> None:
        self.ffmpeg.stdin.close()
        self.ffmpeg.wait()
