import cv2
import numpy as np
import pytest

from birdcam import (
    DEBUG_WINDOW_NAME,
    OUT_H,
    OUT_W,
    configure_debug_window,
    prepare_debug_preview,
)


def test_preview_rotates_90_degrees_clockwise_without_mutating_stream_frame():
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame[:, :, 0] = [[1, 2, 3], [4, 5, 6]]
    original = frame.copy()

    preview = prepare_debug_preview(frame, "clockwise")

    assert preview.shape == (3, 2, 3)
    assert preview[:, :, 0].tolist() == [[4, 1], [5, 2], [6, 3]]
    assert np.array_equal(frame, original)


def test_vertical_stream_canvas_becomes_landscape_preview_by_default():
    frame = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)

    preview = prepare_debug_preview(frame)

    assert preview.shape == (OUT_W, OUT_H, 3)


def test_centered_horizontal_letterbox_becomes_centered_vertical_letterbox():
    frame = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
    content_height = 608
    top = (OUT_H - content_height) // 2
    frame[top:top + content_height] = 255

    preview = prepare_debug_preview(frame, "clockwise")
    occupied = np.argwhere(preview[:, :, 0] > 0)
    left = int(occupied[:, 1].min())
    right = int(occupied[:, 1].max()) + 1

    assert left == preview.shape[1] - right
    assert np.all(preview[:, left:right] == 255)
    assert np.all(preview[:, :left] == 0)
    assert np.all(preview[:, right:] == 0)


@pytest.mark.parametrize("rotation", [None, "none", "off", "0"])
def test_preview_rotation_can_be_disabled(rotation):
    frame = np.zeros((2, 3, 3), dtype=np.uint8)

    assert prepare_debug_preview(frame, rotation) is frame


def test_preview_rejects_unknown_rotation():
    with pytest.raises(ValueError, match="Unsupported preview rotation"):
        prepare_debug_preview(np.zeros((2, 3, 3), dtype=np.uint8), "sideways-ish")


def test_debug_window_is_resizable_and_uses_configured_landscape_size(monkeypatch):
    calls = {}

    def fake_named_window(name, flags):
        calls["named"] = (name, flags)

    def fake_resize_window(name, width, height):
        calls["resized"] = (name, width, height)

    monkeypatch.setattr(cv2, "namedWindow", fake_named_window)
    monkeypatch.setattr(cv2, "resizeWindow", fake_resize_window)

    size = configure_debug_window({"preview_width": 1280, "preview_height": 720})

    assert size == (1280, 720)
    assert calls["named"][0] == DEBUG_WINDOW_NAME
    assert calls["named"][1] & cv2.WINDOW_AUTOSIZE == 0
    assert calls["resized"] == (DEBUG_WINDOW_NAME, 1280, 720)


def test_debug_window_clamps_invalid_dimensions(monkeypatch):
    resized = {}
    monkeypatch.setattr(cv2, "namedWindow", lambda *_args: None)
    monkeypatch.setattr(
        cv2,
        "resizeWindow",
        lambda _name, width, height: resized.update(width=width, height=height),
    )

    size = configure_debug_window({"preview_width": 0, "preview_height": -4})

    assert size == (1, 1)
    assert resized == {"width": 1, "height": 1}
