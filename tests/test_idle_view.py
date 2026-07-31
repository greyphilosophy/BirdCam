import numpy as np

from birdcam import (
    OUT_ASPECT,
    OUT_H,
    OUT_W,
    GuidanceState,
    advance_crop,
    crop_and_scale,
    full_frame_crop,
    overview_crop,
)


def test_full_frame_idle_view_is_letterboxed_with_black_bars():
    frame = np.full((90, 160, 3), 255, dtype=np.uint8)

    output = crop_and_scale(frame, full_frame_crop(160, 90))

    content_height = round(OUT_W * 90 / 160)
    top = (OUT_H - content_height) // 2
    bottom = top + content_height

    assert output.shape == (OUT_H, OUT_W, 3)
    assert np.all(output[:top] == 0)
    assert np.all(output[top:bottom] == 255)
    assert np.all(output[bottom:] == 0)


def test_portrait_tracking_view_still_fills_the_output():
    frame = np.full((160, 90, 3), 127, dtype=np.uint8)

    output = crop_and_scale(frame, full_frame_crop(90, 160))

    assert output.shape == (OUT_H, OUT_W, 3)
    assert np.all(output == 127)


def test_portrait_tracking_transition_preserves_aspect_and_has_no_bars():
    frame = np.full((2160, 3840, 3), 255, dtype=np.uint8)
    current = overview_crop(3840, 2160)
    target = (1700, 700, 540, 960)
    first_advanced = None

    for _ in range(100):
        advanced = advance_crop(
            current,
            target,
            elapsed=0.1,
            frame_w=3840,
            frame_h=2160,
            max_zoom_fraction_per_second=0.35,
            max_pan_fraction_per_second=0.25,
        )
        if first_advanced is None and advanced != current:
            first_advanced = advanced
        assert abs(advanced[2] / advanced[3] - OUT_ASPECT) < 0.002
        current = advanced
        if current == target:
            break

    assert current == target
    assert first_advanced is not None
    assert np.all(crop_and_scale(frame, first_advanced) == 255)


def test_idle_view_is_used_before_any_bird_has_been_seen():
    guidance = GuidanceState()

    crop, mode = guidance.view_for(
        3840,
        2160,
        now=10.0,
        hold_seconds=1.0,
        idle_enabled=True,
        idle_after_seconds=3.0,
    )

    assert crop == full_frame_crop(3840, 2160)
    assert mode == "idle"


def test_view_moves_from_tracking_to_portrait_overview_then_idle():
    guidance = GuidanceState()
    bird_crop = (100, 200, 540, 960)
    guidance.publish(bird_crop, bird_count=1, observed_at=10.0)

    tracking_crop, tracking_mode = guidance.view_for(
        3840,
        2160,
        now=10.5,
        hold_seconds=1.0,
        idle_enabled=True,
        idle_after_seconds=3.0,
    )
    overview, overview_mode = guidance.view_for(
        3840,
        2160,
        now=11.5,
        hold_seconds=1.0,
        idle_enabled=True,
        idle_after_seconds=3.0,
    )
    idle_crop, idle_mode = guidance.view_for(
        3840,
        2160,
        now=13.1,
        hold_seconds=1.0,
        idle_enabled=True,
        idle_after_seconds=3.0,
    )

    assert (tracking_crop, tracking_mode) == (bird_crop, "tracking")
    assert (overview, overview_mode) == (overview_crop(3840, 2160), "overview")
    assert (idle_crop, idle_mode) == (full_frame_crop(3840, 2160), "idle")


def test_idle_view_can_be_disabled():
    guidance = GuidanceState()

    crop, mode = guidance.view_for(
        3840,
        2160,
        now=10.0,
        hold_seconds=1.0,
        idle_enabled=False,
        idle_after_seconds=3.0,
    )

    assert crop == overview_crop(3840, 2160)
    assert mode == "overview"


def test_transition_toward_idle_respects_zoom_speed():
    current = overview_crop(3840, 2160)
    target = full_frame_crop(3840, 2160)

    advanced = advance_crop(
        current,
        target,
        elapsed=0.1,
        frame_w=3840,
        frame_h=2160,
        max_zoom_fraction_per_second=0.35,
        max_pan_fraction_per_second=0.25,
    )

    maximum_width_change = int(current[2] * 0.35 * 0.1)
    assert current[2] < advanced[2] <= current[2] + maximum_width_change
    assert advanced[3] == current[3]
