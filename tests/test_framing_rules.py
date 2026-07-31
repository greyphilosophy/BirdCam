from birdcam import (
    OUT_H,
    OUT_W,
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


def test_widely_separated_birds_expand_to_available_overview():
    birds = [
        (0, 600, 300, 1000),
        (1860, 2800, 2160, 3300),
    ]

    crop = compute_bird_crop(birds, 2160, 3840, padding=200)

    assert crop == (0, 0, 2160, 3840)
    assert all(contains(crop, bird) for bird in birds)


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
