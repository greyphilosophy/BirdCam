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


def test_advance_crop_does_not_move_without_elapsed_time():
    current = overview_crop(3840, 2160)
    target = (1700, 700, 450, 800)
    assert advance_crop(current, target, 0.0, 3840, 2160, 0.25, 0.25) == current


def test_advance_crop_caps_zoom_velocity():
    current = overview_crop(3840, 2160)
    target = (1700, 700, 450, 800)
    advanced = advance_crop(current, target, 0.1, 3840, 2160, 0.25, 10.0)
    assert advanced[2] >= round(current[2] * 0.975) - 1
    assert advanced[2] > target[2]


def test_tight_crop_does_not_force_a_one_pixel_zoom_step():
    current = (100, 100, 40, 71)
    target = (105, 109, 20, 36)

    advanced = advance_crop(current, target, 0.1, 3840, 2160, 0.1, 0.0)

    # The configured cap allows only 0.4 source pixels in this interval. Since
    # crop rectangles are integer-valued, the controller must wait rather than
    # forcing a one-pixel step that would exceed the maximum velocity.
    assert advanced[2] == current[2]


def test_long_stall_does_not_create_catch_up_zoom_jump():
    current = overview_crop(3840, 2160)
    target = (1700, 700, 450, 800)
    normal_step = advance_crop(current, target, 0.1, 3840, 2160, 0.25, 10.0)
    stalled_step = advance_crop(current, target, 5.0, 3840, 2160, 0.25, 10.0)
    assert stalled_step == normal_step


def test_advance_crop_caps_pan_velocity():
    current = (0, 0, 540, 960)
    target = (3300, 1200, 540, 960)
    advanced = advance_crop(current, target, 0.1, 3840, 2160, 10.0, 0.1)
    current_center_x = current[0] + current[2] / 2
    advanced_center_x = advanced[0] + advanced[2] / 2
    assert advanced_center_x - current_center_x <= 3840 * 0.1 * 0.1 + 1
