from birdcam_target_smoothing import SmoothedGuidanceState


def make_state():
    return SmoothedGuidanceState(
        {
            "hold_seconds": 1.0,
            "target_confirmation": {"enabled": True, "required_samples": 2},
            "target_smoothing": {"enabled": False},
        }
    )


def acquire_bird(state):
    bird_crop = (0, 240, 1080, 1920)
    state.publish(bird_crop, bird_count=1, observed_at=0.0)
    state.publish(bird_crop, bird_count=1, observed_at=0.1)
    return bird_crop


def test_single_empty_detection_does_not_pulse_camera_outward():
    state = make_state()
    bird_crop = acquire_bird(state)

    state.publish(None, bird_count=0, observed_at=0.2)

    target, mode = state.view_for(
        3840,
        2160,
        now=0.21,
        hold_seconds=1.0,
        idle_enabled=True,
        idle_after_seconds=3.0,
    )

    assert target == bird_crop
    assert mode == "tracking"


def test_confirmed_empty_detection_stops_chasing_stale_bird_crop():
    state = make_state()
    acquire_bird(state)

    # This reproduces the log pattern where the bird disappears while the
    # renderer is still moving toward a tight bird crop. Two consecutive empty
    # samples confirm loss at the default confirmation setting.
    state.publish(None, bird_count=0, observed_at=0.2)
    state.publish(None, bird_count=0, observed_at=0.3)

    target, mode = state.view_for(
        3840,
        2160,
        now=0.31,
        hold_seconds=1.0,
        idle_enabled=True,
        idle_after_seconds=3.0,
    )

    assert target == (0, 0, 3840, 2160)
    assert mode == "idle"


def test_confirmed_empty_detection_uses_overview_when_idle_disabled():
    state = make_state()
    acquire_bird(state)
    state.publish(None, bird_count=0, observed_at=0.2)
    state.publish(None, bird_count=0, observed_at=0.3)

    target, mode = state.view_for(
        3840,
        2160,
        now=0.31,
        hold_seconds=1.0,
        idle_enabled=False,
    )

    assert target == (1312, 0, 1215, 2160)
    assert mode == "overview"
