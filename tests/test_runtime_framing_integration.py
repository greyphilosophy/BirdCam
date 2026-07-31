import birdcam  # noqa: F401
import birdcam_framing
import birdcam_legacy
import birdcam_optimized


def test_optimized_runtime_uses_framing_layer():
    assert birdcam_legacy.compute_bird_crop is birdcam_framing.compute_bird_crop
    assert birdcam_legacy.advance_crop is birdcam_framing.advance_crop
    assert birdcam_optimized.compute_bird_crop is birdcam_framing.compute_bird_crop
    assert birdcam_optimized.advance_crop is birdcam_framing.advance_crop


def test_runtime_multi_bird_target_expands_and_keeps_output_floor():
    birds = [
        (100, 1200, 400, 1550),
        (1550, 1150, 1900, 1600),
    ]

    crop = birdcam_legacy.compute_bird_crop(
        birds,
        frame_w=2160,
        frame_h=3840,
        padding=100,
    )

    x, y, width, height = crop
    assert width >= birdcam_legacy.OUT_W
    assert height >= birdcam_legacy.OUT_H
    assert x <= min(box[0] for box in birds)
    assert y <= min(box[1] for box in birds)
    assert x + width >= max(box[2] for box in birds)
    assert y + height >= max(box[3] for box in birds)
