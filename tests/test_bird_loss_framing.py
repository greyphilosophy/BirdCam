from birdcam_target_smoothing import SmoothedGuidanceState


def test_empty_detection_immediately_stops_chasing_stale_bird_crop():
    state = SmoothedGuidanceState(
        {
            "hold_seconds": 1.0,
            "target_confirmation": {"enabled": True, "required_samples": 2},
            "target_smoothing": {"enabled": False},
        }
    )
    bird_crop = (0, 240, 1080, 1920)
    state.publish(bird_crop, bird_count=1, observed_at=0.0)
    state.publish(bird_crop, bird_count=1, observed_at=0.1)

    # This reproduces the log pattern where the bird disappears while the
    # renderer is still moving toward a tight bird crop.
    state.publish(None, bird_count=0, observed_at=0.2)

    target, mode = state.view_for(
        3840,
        2160,
        now=0.21,
        hold_seconds=1.0,
        idle_enabled=True,
        idle_after_seconds=3.0,
    )

    assert target == (0, 0, 3840, 2160)
    assert mode == "idle"


def test_empty_detection_uses_portrait_overview_when_idle_view_is_disabled():
    state = SmoothedGuidanceState(
        {
            "target_confirmation": {"enabled": False},
            "target_smoothing": {"enabled": False},
        }
    )
    state.publish((0, 240, 1080, 1920), bird_count=1, observed_at=0.0)
    state.publish(None, bird_count=0, observed_at=0.1)

    target, mode = state.view_for(
        3840,
        2160,
        now=0.11,
        hold_seconds=1.0,
        idle_enabled=False,
    )

    assert target == (1312, 0, 1215, 2160)
    assert mode == "overview"
