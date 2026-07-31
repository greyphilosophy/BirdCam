import threading

import cv2
import numpy as np
import pytest

from birdcam import (
    BIRD_CLASS_ID,
    CaptureWorker,
    LatestFrame,
    detect_birds,
    rotate_frame,
)


def sample_frame():
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    return np.repeat(values[:, :, None], 3, axis=2)


def test_coco_bird_class_is_14():
    assert BIRD_CLASS_ID == 14


def test_detect_birds_keeps_birds_and_rejects_dogs():
    class Boxes:
        xyxy = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)
        cls = np.array([14, 16], dtype=float)

    class Result:
        boxes = Boxes()

    class Model:
        def predict(self, *_args, **_kwargs):
            return [Result()]

    assert detect_birds(Model(), sample_frame()) == [(1, 2, 3, 4)]


def test_rotate_frame_90_degrees_clockwise():
    rotated = rotate_frame(sample_frame(), 90)
    assert rotated.shape == (3, 2, 3)
    assert rotated[:, :, 0].tolist() == [[4, 1], [5, 2], [6, 3]]


def test_rotate_frame_180_degrees():
    rotated = rotate_frame(sample_frame(), 180)
    assert rotated[:, :, 0].tolist() == [[6, 5, 4], [3, 2, 1]]


def test_rotate_frame_270_degrees_clockwise():
    rotated = rotate_frame(sample_frame(), 270)
    assert rotated.shape == (3, 2, 3)
    assert rotated[:, :, 0].tolist() == [[3, 6], [2, 5], [1, 4]]


def test_rotate_frame_accepts_full_turns_and_negative_angles():
    frame = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    assert rotate_frame(frame, 360) is frame
    assert np.array_equal(
        rotate_frame(frame, 450),
        cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE),
    )
    assert np.array_equal(
        rotate_frame(frame, -90),
        cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE),
    )


def test_rotate_frame_rejects_unsupported_angles():
    with pytest.raises(ValueError, match="0, 90, 180, or 270"):
        rotate_frame(sample_frame(), 45)


def test_capture_worker_publishes_native_frame_without_rotation():
    stop_event = threading.Event()
    latest_frame = LatestFrame()
    source = sample_frame()

    class OneFrameCamera:
        def read(self):
            stop_event.set()
            return True, source

        def release(self):
            pass

    worker = CaptureWorker(
        lambda: OneFrameCamera(),
        latest_frame,
        stop_event,
        rotation_degrees=90,
    )
    worker.run()

    snapshot = latest_frame.get()
    assert worker.error is None
    assert snapshot is not None
    assert snapshot.frame is source
    assert snapshot.frame.shape == (2, 3, 3)
