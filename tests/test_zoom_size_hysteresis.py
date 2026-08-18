from birdcam_target_smoothing import SmoothedGuidanceState


def make_state():
    return SmoothedGuidanceState(
        {
            "padding": 200,
            "target_confirmation": {"enabled": False},
            "target_smoothing": {"enabled": False},
        }
    )


def test_single_large_size_outlier_does_not_change_zoom():
    state = make_state()
    stable = (1200, 120, 1080, 1920)
    for index in range(3):
        state.publish(stable, bird_count=1, observed_at=index * 0.1)

    outlier = (0, 0, 3840, 2160)
    state.publish(outlier, bird_count=1, observed_at=0.3)

    target, mode = state.view_for(
        3840,
        2160,
        now=0.3,
        hold_seconds=1.0,
        idle_enabled=True,
        idle_after_seconds=3.0,
    )
    assert target[2:] == stable[2:]
    assert mode == "tracking"


def test_size_outlier_does_not_recenter_to_full_frame():
    state = make_state()
    stable = (0, 120, 1080, 1920)
    for index in range(3):
        state.publish(stable, bird_count=1, observed_at=index * 0.1)

    state.publish((0, 0, 3840, 2160), bird_count=1, observed_at=0.3)

    target, _ = state.view_for(
        3840,
        2160,
        now=0.3,
        hold_seconds=1.0,
        idle_enabled=True,
        idle_after_seconds=3.0,
    )
    assert target == stable


def test_two_large_size_outliers_still_do_not_change_zoom():
    state = make_state()
    stable = (1200, 120, 1080, 1920)
    for index in range(3):
        state.publish(stable, bird_count=1, observed_at=index * 0.1)

    outlier = (0, 0, 3840, 2160)
    state.publish(outlier, bird_count=1, observed_at=0.3)
    state.publish(outlier, bird_count=1, observed_at=0.4)

    target, _ = state.view_for(
        3840,
        2160,
        now=0.4,
        hold_seconds=1.0,
        idle_enabled=True,
        idle_after_seconds=3.0,
    )
    assert target[2:] == stable[2:]


def test_persistent_size_change_eventually_updates_zoom():
    state = make_state()
    stable = (1200, 120, 1080, 1920)
    for index in range(3):
        state.publish(stable, bird_count=1, observed_at=index * 0.1)

    larger = (900, 0, 1800, 2160)
    state.publish(larger, bird_count=1, observed_at=0.3)
    state.publish(larger, bird_count=1, observed_at=0.4)
    state.publish(larger, bird_count=1, observed_at=0.5)

    target, _ = state.view_for(
        3840,
        2160,
        now=0.5,
        hold_seconds=1.0,
        idle_enabled=True,
        idle_after_seconds=3.0,
    )
    assert target[2:] == larger[2:]


def test_padding_is_zoom_size_deadband():
    state = make_state()
    stable = (1200, 120, 1080, 1920)
    for index in range(3):
        state.publish(stable, bird_count=1, observed_at=index * 0.1)

    # Full crop dimensions can move by twice padding before zoom is reframed.
    nearby_size = (1080, 30, 1380, 2100)
    for index in range(3, 8):
        state.publish(nearby_size, bird_count=1, observed_at=index * 0.1)

    target, _ = state.view_for(
        3840,
        2160,
        now=0.7,
        hold_seconds=1.0,
        idle_enabled=True,
        idle_after_seconds=3.0,
    )
    assert target[2:] == stable[2:]


def test_multiple_birds_do_not_reuse_single_bird_zoom_history():
    state = make_state()
    stable = (1200, 120, 1080, 1920)
    for index in range(3):
        state.publish(stable, bird_count=1, observed_at=index * 0.1)

    group = (600, 0, 2200, 2160)
    state.publish(group, bird_count=2, observed_at=0.3)

    target, _ = state.view_for(
        3840,
        2160,
        now=0.3,
        hold_seconds=1.0,
        idle_enabled=True,
        idle_after_seconds=3.0,
    )
    assert target == group
