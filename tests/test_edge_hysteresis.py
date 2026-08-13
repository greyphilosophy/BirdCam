from birdcam import SmoothedGuidanceState, _bird_near_frame_edge


def make_state():
    return SmoothedGuidanceState(
        {
            "padding": 200,
            "hold_seconds": 1.0,
            "target_confirmation": {"enabled": True, "required_samples": 2},
            "target_smoothing": {"enabled": False},
        }
    )


def acquire(state, crop, edge_risk, start=0.0):
    state.set_edge_risk(edge_risk)
    state.publish(crop, bird_count=1, observed_at=start)
    state.set_edge_risk(edge_risk)
    state.publish(crop, bird_count=1, observed_at=start + 0.1)


def test_padding_defines_edge_risk_band():
    frame_w, frame_h, padding = 3840, 2160, 200
    assert not _bird_near_frame_edge([(1000, 700, 1300, 1000)], frame_w, frame_h, padding)
    assert _bird_near_frame_edge([(100, 700, 400, 1000)], frame_w, frame_h, padding)
    assert _bird_near_frame_edge([(3440, 700, 3700, 1000)], frame_w, frame_h, padding)
    assert _bird_near_frame_edge([(1000, 100, 1300, 400)], frame_w, frame_h, padding)
    assert _bird_near_frame_edge([(1000, 1800, 1300, 2050)], frame_w, frame_h, padding)


def test_edge_track_survives_slow_detector_flicker_until_loss_confirms():
    state = make_state()
    crop = (0, 200, 1080, 1920)
    acquire(state, crop, edge_risk=True)

    # Space misses far enough apart that ordinary hold_seconds would expire.
    # Edge hysteresis refreshes the hold until the fourth consecutive miss.
    state.publish(None, bird_count=0, observed_at=0.7)
    state.publish(None, bird_count=0, observed_at=1.3)
    state.publish(None, bird_count=0, observed_at=1.9)

    target, mode = state.view_for(3840, 2160, 1.95, 1.0, idle_enabled=True, idle_after_seconds=3.0)
    assert target == crop
    assert mode == "tracking"

    state.publish(None, bird_count=0, observed_at=2.5)
    target, mode = state.view_for(3840, 2160, 2.51, 1.0, idle_enabled=True, idle_after_seconds=3.0)
    assert target == (0, 0, 3840, 2160)
    assert mode == "idle"


def test_interior_track_keeps_normal_loss_confirmation():
    state = make_state()
    crop = (1200, 120, 1080, 1920)
    acquire(state, crop, edge_risk=False)

    state.publish(None, bird_count=0, observed_at=0.2)
    target, mode = state.view_for(3840, 2160, 0.21, 1.0, idle_enabled=True, idle_after_seconds=3.0)
    assert target == crop
    assert mode == "tracking"

    state.publish(None, bird_count=0, observed_at=0.3)
    target, mode = state.view_for(3840, 2160, 0.31, 1.0, idle_enabled=True, idle_after_seconds=3.0)
    assert target == (0, 0, 3840, 2160)
    assert mode == "idle"


def test_returning_to_safe_region_restores_normal_loss_confirmation():
    state = make_state()
    acquire(state, (0, 200, 1080, 1920), edge_risk=True)

    interior_crop = (1200, 120, 1080, 1920)
    state.set_edge_risk(False)
    state.publish(interior_crop, bird_count=1, observed_at=0.2)
    state.set_edge_risk(False)
    state.publish(interior_crop, bird_count=1, observed_at=0.3)

    state.publish(None, bird_count=0, observed_at=0.4)
    state.publish(None, bird_count=0, observed_at=0.5)
    target, mode = state.view_for(3840, 2160, 0.51, 1.0, idle_enabled=True, idle_after_seconds=3.0)
    assert target == (0, 0, 3840, 2160)
    assert mode == "idle"
