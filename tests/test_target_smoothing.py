from birdcam import BirdCam
from birdcam_target_smoothing import SmoothedGuidanceState


def target(state, now=0.0):
    return state.target_for(2160, 3840, now, hold_seconds=1.0)


def landscape_target(state, now=0.0):
    return state.target_for(3840, 2160, now, hold_seconds=1.0)


def contains(crop, required):
    return (
        crop[0] <= required[0]
        and crop[1] <= required[1]
        and crop[0] + crop[2] >= required[0] + required[2]
        and crop[1] + crop[3] >= required[1] + required[3]
    )


def smoothing_only_config(**overrides):
    smoothing = {"enabled": True, **overrides}
    return {
        "target_confirmation": {"enabled": False},
        "target_smoothing": smoothing,
    }


def test_first_detection_requires_two_agreeing_samples():
    state = SmoothedGuidanceState({})
    detected = (400, 800, 1080, 1920)

    state.publish(detected, bird_count=1, observed_at=0.0)
    assert target(state) != detected

    state.publish((410, 805, 1080, 1920), bird_count=1, observed_at=0.1)
    assert target(state, now=0.1) == (410, 805, 1080, 1920)


def test_conflicting_first_detection_restarts_confirmation():
    state = SmoothedGuidanceState({})
    first = (100, 500, 1080, 1920)
    conflicting = (900, 500, 1080, 1920)

    state.publish(first, bird_count=1, observed_at=0.0)
    state.publish(conflicting, bird_count=1, observed_at=0.1)
    assert target(state, now=0.1) != conflicting

    state.publish((910, 500, 1080, 1920), bird_count=1, observed_at=0.2)
    assert target(state, now=0.2) == (910, 500, 1080, 1920)


def test_large_tracking_jump_requires_confirmation():
    state = SmoothedGuidanceState(
        {
            "target_confirmation": {
                "enabled": True,
                "required_samples": 2,
                "large_center_distance_pixels": 80,
                "large_size_change_fraction": 0.08,
                "agreement_center_distance_pixels": 180,
                "agreement_size_change_fraction": 0.20,
            },
            "target_smoothing": {"enabled": False},
        }
    )
    original = (0, 0, 3840, 2160)
    state.publish(original, bird_count=1, observed_at=0.0)
    state.publish(original, bird_count=1, observed_at=0.1)
    assert landscape_target(state, now=0.1) == original

    proposed = (800, 10, 3013, 2150)
    state.publish(proposed, bird_count=1, observed_at=0.2)
    assert landscape_target(state, now=0.2) == original

    confirmed = (760, 20, 3050, 2140)
    state.publish(confirmed, bird_count=1, observed_at=0.3)
    assert landscape_target(state, now=0.3) == confirmed


def test_full_frame_between_large_candidates_prevents_reframe():
    state = SmoothedGuidanceState(
        {
            "target_confirmation": {"enabled": True},
            "target_smoothing": {"enabled": False},
        }
    )
    full = (0, 0, 3840, 2160)
    state.publish(full, bird_count=1, observed_at=0.0)
    state.publish(full, bird_count=1, observed_at=0.1)

    state.publish((800, 10, 3013, 2150), bird_count=1, observed_at=0.2)
    assert landscape_target(state, now=0.2) == full

    state.publish(full, bird_count=1, observed_at=0.3)
    assert landscape_target(state, now=0.3) == full

    state.publish((522, 84, 3318, 2076), bird_count=1, observed_at=0.4)
    assert landscape_target(state, now=0.4) == full


def test_small_continuous_change_does_not_wait_for_confirmation():
    state = SmoothedGuidanceState(
        {
            "target_confirmation": {
                "enabled": True,
                "large_center_distance_pixels": 80,
                "large_size_change_fraction": 0.08,
            },
            "target_smoothing": {"enabled": False},
        }
    )
    original = (1000, 200, 1200, 1920)
    state.publish(original, bird_count=1, observed_at=0.0)
    state.publish(original, bird_count=1, observed_at=0.1)

    nearby = (1030, 210, 1220, 1910)
    state.publish(nearby, bird_count=1, observed_at=0.2)

    assert landscape_target(state, now=0.2) == nearby


def test_new_additional_bird_requires_confirmation():
    state = SmoothedGuidanceState(
        {
            "target_confirmation": {"enabled": True},
            "target_smoothing": {"enabled": False},
        }
    )
    single = (400, 800, 1080, 1920)
    state.publish(single, bird_count=1, observed_at=0.0)
    state.publish(single, bird_count=1, observed_at=0.1)

    wider_group = (100, 700, 1800, 1920)
    state.publish(wider_group, bird_count=2, observed_at=0.2)
    assert target(state, now=0.2) == single

    state.publish((120, 700, 1780, 1920), bird_count=2, observed_at=0.3)
    assert target(state, now=0.3) == (120, 700, 1780, 1920)


def test_empty_detection_cancels_pending_candidate():
    state = SmoothedGuidanceState({})
    detected = (400, 800, 1080, 1920)

    state.publish(detected, bird_count=1, observed_at=0.0)
    state.publish(None, bird_count=0, observed_at=0.1)
    state.publish(detected, bird_count=1, observed_at=0.2)

    assert target(state, now=0.2) != detected


def test_small_detector_jitter_stays_inside_dead_zone():
    state = SmoothedGuidanceState(
        smoothing_only_config(
            pan_dead_zone_pixels=30,
            zoom_dead_zone_fraction=0.04,
        )
    )
    original = (400, 800, 1080, 1920)
    state.publish(original, bird_count=1, observed_at=0.0)

    state.publish((420, 785, 1100, 1940), bird_count=1, observed_at=0.2)

    assert target(state, now=0.2) == original


def test_meaningful_expansion_contains_current_detection_immediately_after_acceptance():
    state = SmoothedGuidanceState(
        smoothing_only_config(
            center_alpha=0.5,
            size_alpha=0.25,
            pan_dead_zone_pixels=0,
            zoom_dead_zone_fraction=0,
        )
    )
    state.publish((100, 200, 1080, 1920), bird_count=1, observed_at=0.0)
    expanded = (500, 600, 1480, 2320)

    state.publish(expanded, bird_count=1, observed_at=0.2)

    assert contains(target(state, now=0.2), expanded)
    assert target(state, now=0.2) == expanded


def test_safe_contraction_and_recentering_are_smoothed():
    state = SmoothedGuidanceState(
        smoothing_only_config(
            center_alpha=0.5,
            size_alpha=0.25,
            pan_dead_zone_pixels=0,
            zoom_dead_zone_fraction=0,
        )
    )
    state.publish((100, 200, 1480, 2320), bird_count=1, observed_at=0.0)
    smaller = (500, 600, 1080, 1920)

    state.publish(smaller, bird_count=1, observed_at=0.2)

    assert target(state, now=0.2) == (250, 350, 1380, 2220)
    assert contains(target(state, now=0.2), smaller)


def test_brief_empty_detection_remains_continuous_tracking_after_confirmation():
    state = SmoothedGuidanceState(
        {
            "hold_seconds": 1.0,
            "target_confirmation": {"enabled": False},
            "target_smoothing": {"enabled": True},
        }
    )
    state.publish((100, 200, 1480, 2320), bird_count=1, observed_at=0.0)
    state.publish(None, bird_count=0, observed_at=0.2)
    continued = (500, 600, 1080, 1920)

    state.publish(continued, bird_count=1, observed_at=0.4)

    assert target(state, now=0.4) == (184, 284, 1432, 2272)
    assert contains(target(state, now=0.4), continued)


def test_reacquisition_after_hold_period_is_immediate_when_confirmation_disabled():
    state = SmoothedGuidanceState(
        {
            "hold_seconds": 1.0,
            "target_confirmation": {"enabled": False},
            "target_smoothing": {"enabled": True},
        }
    )
    state.publish((100, 200, 1080, 1920), bird_count=1, observed_at=0.0)
    state.publish(None, bird_count=0, observed_at=0.2)
    reacquired = (700, 1000, 1080, 1920)

    state.publish(reacquired, bird_count=1, observed_at=1.2)

    assert target(state, now=1.2) == reacquired


def test_disabled_smoothing_preserves_raw_targets_when_confirmation_disabled():
    state = SmoothedGuidanceState(
        {
            "target_confirmation": {"enabled": False},
            "target_smoothing": {"enabled": False},
        }
    )
    state.publish((100, 200, 1080, 1920), bird_count=1, observed_at=0.0)
    detected = (500, 600, 1480, 2320)

    state.publish(detected, bird_count=1, observed_at=0.2)

    assert target(state, now=0.2) == detected


def test_invalid_numeric_config_falls_back_to_safe_defaults():
    state = SmoothedGuidanceState(
        {
            "target_confirmation": {"enabled": False},
            "target_smoothing": {
                "center_alpha": "invalid",
                "size_alpha": None,
                "pan_dead_zone_pixels": "invalid",
                "zoom_dead_zone_fraction": None,
            },
        }
    )

    state.publish((100, 200, 1480, 2320), bird_count=1, observed_at=0.0)
    state.publish((500, 600, 1080, 1920), bird_count=1, observed_at=0.2)

    assert target(state, now=0.2) == (184, 284, 1432, 2272)


def test_birdcam_uses_smoothed_guidance_without_touching_render_loop():
    camera = BirdCam(
        {
            "camera": {"device": 0},
            "detector": {},
            "tracker": {"target_smoothing": {"enabled": True}},
        }
    )

    assert isinstance(camera.guidance, SmoothedGuidanceState)
