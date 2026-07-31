import types

import numpy as np
import pytest

import birdcam_virtual_camera as virtual_camera


class Logger:
    def info(self, *args, **kwargs):
        pass


class FakeCamera:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.device = "Fake Virtual Camera"
        self.frames = []
        self.closed = False

    def send(self, frame):
        self.frames.append(frame)

    def close(self):
        self.closed = True


def fake_module():
    return types.SimpleNamespace(
        PixelFormat=types.SimpleNamespace(BGR="BGR"),
        Camera=FakeCamera,
    )


def test_disabled_virtual_camera_does_not_import_backend(monkeypatch):
    output = virtual_camera.VirtualCameraOutput({}, 1080, 1920, 60, Logger())
    monkeypatch.setattr(
        virtual_camera.importlib,
        "import_module",
        lambda name: pytest.fail("backend should not be imported"),
    )

    output.start()

    assert output.camera is None


def test_virtual_camera_sends_bgr_frame_without_copy(monkeypatch):
    module = fake_module()
    monkeypatch.setattr(virtual_camera.importlib, "import_module", lambda name: module)
    output = virtual_camera.VirtualCameraOutput(
        {"enabled": True, "backend": "obs"},
        1080,
        1920,
        60,
        Logger(),
    )
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)

    output.start()
    output.send_frame(frame)

    assert output.camera.kwargs == {
        "width": 1080,
        "height": 1920,
        "fps": 60.0,
        "fmt": "BGR",
        "backend": "obs",
    }
    assert output.camera.frames == [frame]


def test_composite_output_fans_out_and_stops_in_reverse_order():
    events = []

    class Sink:
        def __init__(self, name):
            self.name = name

        def send_frame(self, frame):
            events.append(("send", self.name, frame))

        def stop(self):
            events.append(("stop", self.name))

    output = virtual_camera.CompositeOutput([Sink("stream"), None, Sink("camera")])

    output.send_frame("frame")
    output.stop()

    assert events == [
        ("send", "stream", "frame"),
        ("send", "camera", "frame"),
        ("stop", "camera"),
        ("stop", "stream"),
    ]
