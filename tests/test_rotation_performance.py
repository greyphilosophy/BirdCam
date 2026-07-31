import cv2
import numpy as np
import pytest

from birdcam import (
    OUT_H,
    OUT_W,
    crop_and_scale,
    crop_and_scale_rotated,
    native_crop_for_rotated_crop,
    rotate_frame,
    rotated_frame_size,
)


def test_rotated_frame_size_swaps_quarter_turn_dimensions():
    assert rotated_frame_size(3840, 2160, 90) == (2160, 3840)
    assert rotated_frame_size(3840, 2160, 270) == (2160, 3840)
    assert rotated_frame_size(3840, 2160, 0) == (3840, 2160)
    assert rotated_frame_size(3840, 2160, 180) == (3840, 2160)


def test_rotated_frame_size_rejects_unsupported_angle():
    with pytest.raises(ValueError, match="0, 90, 180, or 270"):
        rotated_frame_size(3840, 2160, 45)


def test_clockwise_crop_maps_back_to_native_coordinates():
    assert native_crop_for_rotated_crop(
        (100, 300, 400, 800),
        3840,
        2160,
        90,
    ) == (300, 1660, 800, 400)


def test_counterclockwise_crop_maps_back_to_native_coordinates():
    assert native_crop_for_rotated_crop(
        (100, 300, 400, 800),
        3840,
        2160,
        270,
    ) == (2740, 100, 800, 400)


def test_optimized_renderer_matches_rotate_then_crop_reference():
    frame = np.arange(60 * 100 * 3, dtype=np.uint8).reshape(60, 100, 3)
    rotated = rotate_frame(frame, 90)
    crop = (5, 10, 40, 70)

    expected = crop_and_scale(rotated, crop)
    actual = crop_and_scale_rotated(frame, crop, 90)

    assert actual.shape == expected.shape == (OUT_H, OUT_W, 3)
    difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    assert np.percentile(difference, 99.9) <= 1
    assert difference.max() <= 2


def test_full_frame_resizes_before_clockwise_rotation(monkeypatch):
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    calls = {}
    real_resize = cv2.resize
    real_rotate = cv2.rotate

    def recording_resize(image, size, interpolation):
        calls["resize_input"] = image.shape
        calls["resize_size"] = size
        return real_resize(image, size, interpolation=interpolation)

    def recording_rotate(image, code):
        calls["rotate_input"] = image.shape
        return real_rotate(image, code)

    monkeypatch.setattr(cv2, "resize", recording_resize)
    monkeypatch.setattr(cv2, "rotate", recording_rotate)

    output = crop_and_scale_rotated(
        frame,
        (0, 0, 2160, 3840),
        90,
    )

    assert output.shape == (1920, 1080, 3)
    assert calls["resize_input"] == (2160, 3840, 3)
    assert calls["resize_size"] == (1920, 1080)
    assert calls["rotate_input"] == (1080, 1920, 3)


def test_zero_rotation_matches_existing_renderer():
    frame = np.arange(60 * 100 * 3, dtype=np.uint8).reshape(60, 100, 3)
    crop = (10, 5, 50, 50)

    assert np.array_equal(
        crop_and_scale_rotated(frame, crop, 0),
        crop_and_scale(frame, crop),
    )
