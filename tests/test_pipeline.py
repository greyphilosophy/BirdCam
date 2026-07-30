import numpy as np

from birdcam import GuidanceState, LatestFrame, overview_crop


def test_latest_frame_keeps_only_newest_snapshot():
    frames = LatestFrame()
    first = np.zeros((2, 2, 3), dtype=np.uint8)
    second = np.ones((2, 2, 3), dtype=np.uint8)

    frames.publish(first, captured_at=1.0)
    frames.publish(second, captured_at=2.0)
    snapshot = frames.get()

    assert snapshot.sequence == 2
    assert snapshot.captured_at == 2.0
    assert snapshot.frame is second


def test_latest_frame_preserves_explicit_zero_timestamp():
    frames = LatestFrame()
    frames.publish(np.zeros((2, 2, 3), dtype=np.uint8), captured_at=0.0)

    assert frames.get().captured_at == 0.0


def test_wait_for_newer_does_not_return_same_frame():
    frames = LatestFrame()
    frames.publish(np.zeros((2, 2, 3), dtype=np.uint8), captured_at=1.0)

    assert frames.wait_for_newer(1, timeout=0.001) is None


def test_guidance_holds_last_bird_crop_then_returns_to_overview():
    guidance = GuidanceState()
    bird_crop = (100, 200, 540, 960)
    guidance.publish(bird_crop, bird_count=1, observed_at=10.0, published_at=10.0)
    guidance.publish(None, bird_count=0, observed_at=10.2, published_at=10.2)

    assert guidance.target_for(3840, 2160, now=10.8, hold_seconds=1.0) == bird_crop
    assert guidance.target_for(3840, 2160, now=11.1, hold_seconds=1.0) == overview_crop(3840, 2160)


def test_slow_result_expires_from_source_frame_time():
    guidance = GuidanceState()
    bird_crop = (100, 200, 540, 960)
    guidance.publish(bird_crop, bird_count=1, observed_at=5.0, published_at=10.0)

    assert guidance.target_for(3840, 2160, now=10.5, hold_seconds=1.0) == overview_crop(3840, 2160)
