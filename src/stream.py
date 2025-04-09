import subprocess


class YouTubeStream:
    def __init__(self, token: str):
        self.token = token

        self.ffmpeg = None

    def start(self) -> None:
        # fmt: off
        # self.ffmpeg = subprocess.Popen(
        #     [
        #         "ffmpeg",
        #         "-f", "rawvideo",
        #         "-pix_fmt", "bgr24",
        #         "-s", "1920x1080",
        #         "-r", "30",
        #         "-i", "-",
        #         "-f", "lavfi",
        #         "-i", "anullsrc",  # Silent audio
        #         "-c:v", "libx264",
        #         "-preset", "ultrafast",
        #         "-tune", "zerolatency",
        #         "-b:v", "3000k",
        #         "-bufsize", "6000k",
        #         "-pix_fmt", "yuv420p",
        #         "-g", "60",
        #         "-keyint_min", "30",
        #         "-crf", "25",
        #         "-c:a", "aac",
        #         "-b:a", "128k",
        #         "-ar", "44100",
        #         "-f", "flv",
        #         f"rtmp://a.rtmp.youtube.com/live2/{self.token}"
        #     ],
        #     stdin=subprocess.PIPE,
        # )
        self.ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-i", "pipe:",
                "-c:v", "h264_nvenc",
                "-c:a", "aac",
                "-ar", "44100",
                "-b:a", "128k",
                "-f", "flv",
                "-hwaccel", "cuda",
                f"rtmp://a.rtmp.youtube.com/live2/{self.token}"
            ],
            stdout=subprocess.PIPE,
            )
        # fmt: on

    def stop(self) -> None:
        self.ffmpeg.stdin.close()
        self.ffmpeg.wait()
