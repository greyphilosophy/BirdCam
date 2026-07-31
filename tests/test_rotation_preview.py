import threading

import cv2
import numpy as np
import pytest

from birdcam import CaptureWorker, LatestFrame, fit_preview, rotate_frame


def test_rotate_frame_90_degrees_clockwise():
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    frame = np.repeat(values[:, :, None], 3, axis=2)

    rotated = rotate_frame(frame, 90)

    assert rotated.shape == (3, 2, 3)
    assert rotated[:, :, 0].tolist() == [[4, 1], [5, 2], [6, 3]]


def test_rotate_frame_180_degrees():
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    frame = np.repeat(values[:, :, None], 3, axis=2)

    rotated = rotate_frame(frame, 180)

    assert rotated[:, :, 0].tolist() == [[6, 5, 4], [3, 2, 1]]


def test_rotate_frame_270_degrees_clockwise():
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    frame = np.repeat(values[:, :, None], 3, axis=2)

    rotated = rotate_frame(frame, 270)

    assert rotated.shape == (3, 2, 3)
    assert rotated[:, :, 0].tolist() == [[3, 6], [2, 5], [1, 4]]


def test_rotate_frame_accepts_full_turns_and_negative_angles():
    frame = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)

    assert rotate_frame(frame, 360) is frame
    assert np.array_equal(rotate_frame(frame, 450), cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE))
    assert np.array_equal(
        rotate_frame(frame, -90),
        cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE),
    )


def test_rotate_frame_rejects_unsupported_angles():
    frame = np.zeros((2, 3, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="0, 90, 180, or 270"):
        rotate_frame(frame, 45)


def test_capture_worker_publishes_rotated_frame():
    stop_event = threading.Event()
    latest_frame = LatestFrame()
    values = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    source = np.repeat(values[:, :, None], 3, axis=2)

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
    assert snapshot.frame[:, :, 0].tolist() == [[4, 1], [5, 2], [6, 3]]


def test_fit_preview_preserves_portrait_aspect_within_bounds():
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)

    preview = fit_preview(frame, max_width=960, max_height=900)

    assert preview.shape == (900, 506, 3)
    assert preview.shape[1] <= 960
    assert preview.shape[0] <= 900
    assert abs(preview.shape[1] / preview.shape[0] - 1080 / 1920) < 0.002


def test_fit_preview_preserves_landscape_aspect_when_width_limited():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    preview = fit_preview(frame, max_width=900, max_height=960)

    assert preview.shape == (506, 900, 3)
    assert preview.shape[1] <= 900
    assert preview.shape[0] <= 960
    assert abs(preview.shape[1] / preview.shape[0] - 1920 / 1080) < 0.002


def test_fit_preview_does_not_enlarge_small_frames():
    frame = np.zeros((200, 100, 3), dtype=np.uint8)

    assert fit_preview(frame, max_width=960, max_height=900) is frame
