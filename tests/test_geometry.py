from birdcam import OUT_ASPECT, advance_crop, compute_bird_crop, overview_crop


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


def test_wide_group_falls_back_to_widest_portrait_overview():
    crop = compute_bird_crop([(0, 900, 3800, 1200)], 3840, 2160, padding=0)

    assert crop == overview_crop(3840, 2160)


def test_overview_crop_is_centered_and_vertical():
    x, y, width, height = overview_crop(3840, 2160)

    assert abs(width / height - OUT_ASPECT) < 0.002
    assert x == (3840 - width) // 2
    assert y == 0


def test_advance_crop_caps_zoom_velocity():
    current = overview_crop(3840, 2160)
    target = (1700, 700, 450, 800)

    advanced = advance_crop(
        current,
        target,
        elapsed=0.1,
        frame_w=3840,
        frame_h=2160,
        max_zoom_fraction_per_second=0.25,
        max_pan_fraction_per_second=10.0,
    )

    # At 25% per second for 0.1 seconds, width may shrink by at most 2.5%.
    assert advanced[2] >= round(current[2] * 0.975) - 1
    assert advanced[2] > target[2]


def test_advance_crop_caps_pan_velocity():
    current = (0, 0, 540, 960)
    target = (3300, 1200, 540, 960)

    advanced = advance_crop(
        current,
        target,
        elapsed=0.1,
        frame_w=3840,
        frame_h=2160,
        max_zoom_fraction_per_second=10.0,
        max_pan_fraction_per_second=0.1,
    )

    current_center_x = current[0] + current[2] / 2
    advanced_center_x = advanced[0] + advanced[2] / 2
    assert advanced_center_x - current_center_x <= 3840 * 0.1 * 0.1 + 1
