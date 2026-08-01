from birdcam import BirdCam
from birdcam_target_smoothing import SmoothedGuidanceState


def target(state, now=0.0):
    return state.target_for(2160, 3840, now, hold_seconds=1.0)


def test_first_detection_is_accepted_immediately():
    state = SmoothedGuidanceState({"target_smoothing": {"enabled": True}})
    detected = (400, 800, 1080, 1920)

    state.publish(detected, bird_count=1, observed_at=0.0)

    assert target(state) == detected


def test_first_detection_is_valid_when_immediate_acquisition_is_disabled():
    state = SmoothedGuidanceState(
        {"target_smoothing": {"enabled": True, "immediate_on_acquire": False}}
    )
    detected = (400, 800, 1080, 1920)

    state.publish(detected, bird_count=1, observed_at=0.0)

    assert target(state) == detected


def test_new_additional_bird_is_accepted_immediately():
    state = SmoothedGuidanceState({"target_smoothing": {"enabled": True}})
    state.publish((400, 800, 1080, 1920), bird_count=1, observed_at=0.0)
    wider_group = (100, 700, 1800, 1920)

    state.publish(wider_group, bird_count=2, observed_at=0.2)

    assert target(state, now=0.2) == wider_group


def test_small_detector_jitter_stays_inside_dead_zone():
    state = SmoothedGuidanceState(
        {
            "target_smoothing": {
                "enabled": True,
                "pan_dead_zone_pixels": 30,
                "zoom_dead_zone_fraction": 0.04,
            }
        }
    )
    original = (400, 800, 1080, 1920)
    state.publish(original, bird_count=1, observed_at=0.0)

    state.publish((420, 785, 1100, 1940), bird_count=1, observed_at=0.2)

    assert target(state, now=0.2) == original


def test_meaningful_tracking_changes_are_smoothed():
    state = SmoothedGuidanceState(
        {
            "target_smoothing": {
                "enabled": True,
                "center_alpha": 0.5,
                "size_alpha": 0.25,
                "pan_dead_zone_pixels": 0,
                "zoom_dead_zone_fraction": 0,
            }
        }
    )
    state.publish((100, 200, 1080, 1920), bird_count=1, observed_at=0.0)

    state.publish((500, 600, 1480, 2320), bird_count=1, observed_at=0.2)

    assert target(state, now=0.2) == (350, 450, 1180, 2020)


def test_reacquisition_after_empty_detection_is_immediate():
    state = SmoothedGuidanceState({"target_smoothing": {"enabled": True}})
    state.publish((100, 200, 1080, 1920), bird_count=1, observed_at=0.0)
    state.publish(None, bird_count=0, observed_at=0.2)
    reacquired = (700, 1000, 1080, 1920)

    state.publish(reacquired, bird_count=1, observed_at=0.4)

    assert target(state, now=0.4) == reacquired


def test_disabled_smoothing_preserves_raw_targets():
    state = SmoothedGuidanceState({"target_smoothing": {"enabled": False}})
    state.publish((100, 200, 1080, 1920), bird_count=1, observed_at=0.0)
    detected = (500, 600, 1480, 2320)

    state.publish(detected, bird_count=1, observed_at=0.2)

    assert target(state, now=0.2) == detected


def test_invalid_numeric_config_falls_back_to_safe_defaults():
    state = SmoothedGuidanceState(
        {
            "target_smoothing": {
                "center_alpha": "invalid",
                "size_alpha": None,
                "pan_dead_zone_pixels": "invalid",
                "zoom_dead_zone_fraction": None,
            }
        }
    )

    state.publish((100, 200, 1080, 1920), bird_count=1, observed_at=0.0)
    state.publish((500, 600, 1480, 2320), bird_count=1, observed_at=0.2)

    assert target(state, now=0.2) == (256, 356, 1128, 1968)


def test_birdcam_uses_smoothed_guidance_without_touching_render_loop():
    camera = BirdCam(
        {
            "camera": {"device": 0},
            "detector": {},
            "tracker": {"target_smoothing": {"enabled": True}},
        }
    )

    assert isinstance(camera.guidance, SmoothedGuidanceState)
