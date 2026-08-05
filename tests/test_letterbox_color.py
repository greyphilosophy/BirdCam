import numpy as np

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
