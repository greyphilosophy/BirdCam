from pathlib import Path

import cv2

from birdcam import OUT_H, OUT_W, crop_and_scale, overview_crop
from photo_source import PhotoSequenceCapture


PHOTO_DIR = Path(__file__).resolve().parents[1] / "test_photos"
PHOTO_PATTERN = str(PHOTO_DIR / "P*.JPG")
EXPECTED_NAMES = [
    *(f"P1200{number}.JPG" for number in range(237, 242)),
    *(f"P1200{number}.JPG" for number in range(244, 265)),
]


def test_full_resolution_photo_fixture_set_is_complete():
    photos = sorted(PHOTO_DIR.glob("P*.JPG"))

    assert [photo.name for photo in photos] == EXPECTED_NAMES
    assert len(photos) == 26


def test_representative_fixtures_are_original_high_resolution_photos():
    photos = sorted(PHOTO_DIR.glob("P*.JPG"))

    for photo in (photos[0], photos[len(photos) // 2], photos[-1]):
        frame = cv2.imread(str(photo), cv2.IMREAD_COLOR)
        assert frame is not None, f"Unable to decode {photo.name}"
        height, width = frame.shape[:2]
        assert width > height
        assert width * height >= 12_000_000


def test_real_photos_emulate_configured_4k_camera_frames():
    source = PhotoSequenceCapture(
        PHOTO_PATTERN,
        fps=60,
        seconds_per_photo=1,
        realtime=False,
        width=3840,
        height=2160,
    )

    ok, frame = source.read()

    assert ok
    assert frame.shape == (2160, 3840, 3)
    assert source.get(cv2.CAP_PROP_FRAME_WIDTH) == 3840
    assert source.get(cv2.CAP_PROP_FRAME_HEIGHT) == 2160
    assert source.get(cv2.CAP_PROP_FPS) == 60


def test_real_photo_reaches_vertical_output_pipeline():
    source = PhotoSequenceCapture(
        PHOTO_PATTERN,
        fps=60,
        realtime=False,
        width=3840,
        height=2160,
    )
    ok, frame = source.read()
    assert ok

    output = crop_and_scale(frame, overview_crop(3840, 2160))

    assert output.shape == (OUT_H, OUT_W, 3)
    assert output.std() > 1.0
