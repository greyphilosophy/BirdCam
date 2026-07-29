import numpy as np

from birdcam import OUT_ASPECT, compute_bird_crop, full_frame_view


def test_compute_bird_crop_preserves_vertical_aspect_ratio():
    crop = compute_bird_crop([(1000, 800, 1200, 1000)], 3840, 2160, padding=200)

    assert crop is not None
    _, _, width, height = crop
    assert abs(width / height - OUT_ASPECT) < 0.002


def test_compute_bird_crop_stays_inside_frame_at_edge():
    crop = compute_bird_crop([(0, 0, 100, 100)], 3840, 2160, padding=200)

    assert crop is not None
    x, y, width, height = crop
    assert x >= 0
    assert y >= 0
    assert x + width <= 3840
    assert y + height <= 2160


def test_compute_bird_crop_returns_none_when_group_cannot_fit_portrait_crop():
    crop = compute_bird_crop([(0, 900, 3800, 1200)], 3840, 2160, padding=0)

    assert crop is None


def test_full_frame_view_letterboxes_arbitrary_source_size():
    frame = np.full((720, 1280, 3), 255, dtype=np.uint8)

    output = full_frame_view(frame)

    assert output.shape == (1920, 1080, 3)
    assert output[960, 540].tolist() == [255, 255, 255]
    assert output[0, 0].tolist() == [0, 0, 0]
