import numpy as np
import pytest

import birdcam
import birdcam_legacy as legacy
from birdcam_letterbox import LETTERBOX_GRAY, apply_dark_gray_letterbox


def test_horizontal_letterbox_bars_are_dark_gray_without_touching_image():
    frame = np.full((legacy.OUT_H, legacy.OUT_W, 3), 200, dtype=np.uint8)
    crop = (0, 0, 1920, 1080)

    output = apply_dark_gray_letterbox(frame, crop)

    render_height = int(round(1080 * min(legacy.OUT_W / 1920, legacy.OUT_H / 1080)))
    offset_y = (legacy.OUT_H - render_height) // 2
    assert np.all(output[:offset_y] == LETTERBOX_GRAY)
    assert np.all(output[offset_y + render_height:] == LETTERBOX_GRAY)
    assert np.all(output[offset_y:offset_y + render_height] == 200)


def test_vertical_letterbox_bars_are_dark_gray_without_touching_image():
    frame = np.full((legacy.OUT_H, legacy.OUT_W, 3), 200, dtype=np.uint8)
    crop = (0, 0, 500, 1920)

    output = apply_dark_gray_letterbox(frame, crop)

    offset_x = (legacy.OUT_W - 500) // 2
    assert np.all(output[:, :offset_x] == LETTERBOX_GRAY)
    assert np.all(output[:, offset_x + 500:] == LETTERBOX_GRAY)
    assert np.all(output[:, offset_x:offset_x + 500] == 200)


def test_matching_portrait_output_is_unchanged():
    frame = np.zeros((legacy.OUT_H, legacy.OUT_W, 3), dtype=np.uint8)

    output = apply_dark_gray_letterbox(frame, (0, 0, 1080, 1920))

    assert output is frame
    assert np.all(output == 0)


def assert_production_letterbox_preserves_video(
    frame,
    crop,
    degrees,
    require_bars=False,
):
    raw = birdcam._optimized_crop_and_scale_rotated(frame, crop, degrees)
    output = birdcam._crop_and_scale_with_dark_gray_letterbox(frame, crop, degrees)

    video_mask = np.any(raw != 0, axis=2)
    bar_mask = ~video_mask

    assert np.any(video_mask)
    assert np.array_equal(output[video_mask], raw[video_mask])
    if require_bars:
        assert np.any(bar_mask)
    if np.any(bar_mask):
        assert np.all(output[bar_mask] == LETTERBOX_GRAY)


@pytest.mark.parametrize("degrees", [0, 90, 180, 270])
def test_production_letterbox_never_overwrites_rendered_video_pixels(degrees):
    """Gray recoloring must never change pixels emitted by the video renderer."""
    native_h, native_w = 901, 1601
    frame = np.full((native_h, native_w, 3), 123, dtype=np.uint8)
    if degrees in {90, 270}:
        crop = (0, 0, native_h, native_w)
    else:
        crop = (0, 0, native_w, native_h)

    assert_production_letterbox_preserves_video(frame, crop, degrees)


@pytest.mark.parametrize(
    "crop",
    [
        (385, 216, 3455, 1944),
        (1519, 195, 1520, 1941),
        (1169, 0, 1502, 2160),
    ],
)
def test_production_letterbox_preserves_video_for_logged_crop_shapes(crop):
    """Exercise crop geometries observed in production logs."""
    frame = np.full((2160, 3840, 3), 137, dtype=np.uint8)

    assert_production_letterbox_preserves_video(
        frame,
        crop,
        0,
        require_bars=True,
    )


def test_production_letterbox_preserves_video_at_odd_rounding_boundary():
    """Odd render dimensions must not make a bar consume an adjacent video row."""
    native_h, native_w = 1601, 1001
    frame = np.full((native_h, native_w, 3), 173, dtype=np.uint8)
    crop = (0, 0, native_w, native_h)

    assert_production_letterbox_preserves_video(
        frame,
        crop,
        0,
        require_bars=True,
    )
