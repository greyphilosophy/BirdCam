from birdcam import OUT_H, OUT_W
from birdcam_framing import (
    advance_crop,
    compute_bird_crop,
    ensure_minimum_output_crop,
)


def contains(crop, box):
    x, y, width, height = crop
    x1, y1, x2, y2 = box
    return x <= x1 and y <= y1 and x + width >= x2 and y + height >= y2


def test_single_bird_crop_never_goes_below_output_resolution():
    crop = compute_bird_crop([(1000, 1800, 1100, 1900)], 2160, 3840, padding=0)

    assert crop[2] >= OUT_W
    assert crop[3] >= OUT_H


def test_multi_bird_crop_contains_every_detected_bird():
    birds = [
        (150, 1200, 450, 1550),
        (1550, 1150, 1900, 1600),
    ]

    crop = compute_bird_crop(birds, 2160, 3840, padding=100)

    assert all(contains(crop, bird) for bird in birds)
    assert crop[2] > OUT_W
    assert crop[3] > OUT_H


def test_widely_separated_birds_use_minimum_enclosing_crop():
    birds = [
        (50, 1400, 250, 1700),
        (1910, 1500, 2110, 1800),
    ]

    crop = compute_bird_crop(birds, 2160, 3840, padding=200)

    assert crop == (0, 640, 2160, OUT_H)
    assert crop != (0, 0, 2160, 3840)
    assert all(contains(crop, bird) for bird in birds)


def test_multi_bird_crop_at_exact_frame_limit_preserves_both_birds():
    birds = [
        (100, 1400, 300, 1700),
        (1860, 1500, 2060, 1800),
    ]

    crop = compute_bird_crop(birds, 2160, 3840, padding=100)

    assert crop == (0, 640, 2160, OUT_H)
    assert crop != (0, 0, 2160, 3840)
    assert all(contains(crop, bird) for bird in birds)


def test_enclosing_crop_clamps_at_frame_edge_without_losing_birds():
    birds = [
        (0, 100, 180, 400),
        (1980, 200, 2160, 500),
    ]

    crop = compute_bird_crop(birds, 2160, 3840, padding=200)

    assert crop == (0, 0, 2160, OUT_H)
    assert all(contains(crop, bird) for bird in birds)


def test_no_birds_still_produces_no_tracking_crop():
    assert compute_bird_crop([], 2160, 3840, padding=200) is None


def test_minimum_crop_preserves_center_when_expanding():
    crop = ensure_minimum_output_crop((800, 1500, 540, 960), 2160, 3840)

    assert crop[2:] == (OUT_W, OUT_H)
    assert crop[0] + crop[2] / 2 == 800 + 540 / 2
    assert crop[1] + crop[3] / 2 == 1500 + 960 / 2


def test_advance_crop_never_emits_sub_output_dimensions():
    advanced = advance_crop(
        (800, 1500, 540, 960),
        (900, 1600, 400, 711),
        elapsed=0.1,
        frame_w=2160,
        frame_h=3840,
        max_zoom_fraction_per_second=0.35,
        max_pan_fraction_per_second=0.25,
    )

    assert advanced[2] >= OUT_W
    assert advanced[3] >= OUT_H
