class Camera:
    """Base Camera Class."""

    def __init__(self, event_stop) -> None:

        self.event_stop = event_stop

    """ def start(self) -> None:
        raise NotImplementedError("Function start() must be implemented in subclass.") """

    # def stop(self):
    #     self.running = False

    # def get_frame(self):
    #     raise NotImplementedError("Function get_frame() must be implemented in subclass.")
