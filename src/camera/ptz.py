import NDIlib as ndi
import numpy as np


class PTZCamera:
    """PTZ Camera class."""

    def __init__(self, src: str, idx: int) -> None:

        self.idx = idx
        self.receiver = self.__create_receiver(src)

    def __create_receiver(self, src):

        ndi_recv_create = ndi.RecvCreateV3()
        ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
        receiver = ndi.recv_create_v3(ndi_recv_create)
        if receiver is None:
            raise RuntimeError("Failed to create NDI receiver")
        ndi.recv_connect(receiver, src)

        return receiver

    def get_frame(self):

        t, v, _, _ = ndi.recv_capture_v3(self.receiver, 1000)
        frame = None
        if t == ndi.FRAME_TYPE_VIDEO:
            frame = np.copy(v.data[:, :, :3])
            ndi.recv_free_video_v2(self.receiver, v)

        return frame, t

    def move(self) -> None:
        pass

    def __ease_in_out(self, t):
        """
        Cubic ease-in-out function.

        t: Normalized time between 0 and 1.

        Returns the adjusted value based on ease-in-out.
        """

        if t < 0.5:
            return 4 * t**3
        return 1 - (-2 * t + 2) ** 3 / 2
