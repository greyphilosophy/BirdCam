"""Direct virtual-camera output for BirdCam."""

import importlib


class VirtualCameraOutput:
    """Publish rendered BGR frames through a pyvirtualcam backend."""

    def __init__(self, config, width, height, fps, logger):
        self.config = config or {}
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.logger = logger
        self.camera = None

    @property
    def enabled(self):
        return bool(self.config.get("enabled", False))

    def start(self):
        if not self.enabled:
            return
        try:
            pyvirtualcam = importlib.import_module("pyvirtualcam")
        except ImportError as exc:
            raise RuntimeError(
                "virtual_camera.enabled is true, but pyvirtualcam is not installed"
            ) from exc

        kwargs = {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "fmt": pyvirtualcam.PixelFormat.BGR,
            "backend": self.config.get("backend", "obs"),
        }
        if self.config.get("device"):
            kwargs["device"] = self.config["device"]
        self.camera = pyvirtualcam.Camera(**kwargs)
        self.logger.info(
            "Virtual camera started: %s (%dx%d at %.1f fps)",
            getattr(self.camera, "device", "virtual camera"),
            self.width,
            self.height,
            self.fps,
        )

    def send_frame(self, frame):
        if self.camera is not None:
            self.camera.send(frame)

    def stop(self):
        camera, self.camera = self.camera, None
        if camera is not None:
            camera.close()


class CompositeOutput:
    """Fan one rendered frame out to multiple output sinks."""

    def __init__(self, outputs):
        self.outputs = [output for output in outputs if output is not None]

    def send_frame(self, frame):
        for output in self.outputs:
            output.send_frame(frame)

    def stop(self):
        first_error = None
        for output in reversed(self.outputs):
            try:
                output.stop()
            except Exception as exc:  # Keep closing the remaining outputs.
                first_error = first_error or exc
        if first_error is not None:
            raise first_error
