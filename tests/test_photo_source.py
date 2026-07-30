import cv2
import numpy as np

from photo_source import PhotoSequenceCapture


def write_photo(path, value, width=64, height=36):
    frame = np.full((height, width, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), frame)


def test_photo_source_holds_each_photo_for_configured_frames(tmp_path):
    write_photo(tmp_path / "01.jpg", 20)
    write_photo(tmp_path / "02.jpg", 220)
    source = PhotoSequenceCapture(
        str(tmp_path / "*.jpg"), fps=4, seconds_per_photo=0.5, realtime=False
    )

    values = []
    for _ in range(5):
        ok, frame = source.read()
        assert ok
        values.append(int(frame.mean()))

    assert values[0] < 50
    assert values[1] < 50
    assert values[2] > 180
    assert values[3] > 180
    assert values[4] < 50


def test_photo_source_reports_dimensions_and_fps(tmp_path):
    write_photo(tmp_path / "bird.jpg", 100, width=128, height=72)
    source = PhotoSequenceCapture(str(tmp_path / "*.jpg"), fps=60, realtime=False)

    assert source.isOpened()
    assert source.get(cv2.CAP_PROP_FRAME_WIDTH) == 128
    assert source.get(cv2.CAP_PROP_FRAME_HEIGHT) == 72
    assert source.get(cv2.CAP_PROP_FPS) == 60


def test_photo_source_can_stop_instead_of_looping(tmp_path):
    write_photo(tmp_path / "bird.jpg", 100)
    source = PhotoSequenceCapture(
        str(tmp_path / "*.jpg"), fps=1, seconds_per_photo=1, loop=False, realtime=False
    )

    ok, frame = source.read()
    assert ok and frame is not None
    assert not source.isOpened()
    assert source.read() == (False, None)


def test_photo_source_returns_independent_frame_copies(tmp_path):
    write_photo(tmp_path / "bird.jpg", 100)
    source = PhotoSequenceCapture(str(tmp_path / "*.jpg"), realtime=False)

    _, first = source.read()
    first[:] = 0
    _, second = source.read()

    assert second.mean() > 50


def test_photo_source_paces_frames_at_configured_rate(tmp_path):
    write_photo(tmp_path / "bird.jpg", 100)
    now = [10.0]
    sleeps = []

    def clock():
        return now[0]

    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    source = PhotoSequenceCapture(
        str(tmp_path / "*.jpg"), fps=60, realtime=True, clock=clock, sleep=sleep
    )
    source.read()
    source.read()
    source.read()

    assert len(sleeps) == 2
    assert abs(sleeps[0] - 1 / 60) < 1e-9
    assert abs(sleeps[1] - 1 / 60) < 1e-9


def test_photo_source_does_not_burst_after_clock_stall(tmp_path):
    write_photo(tmp_path / "bird.jpg", 100)
    now = [10.0]
    sleeps = []

    def clock():
        return now[0]

    def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    source = PhotoSequenceCapture(
        str(tmp_path / "*.jpg"), fps=60, realtime=True, clock=clock, sleep=sleep
    )
    source.read()
    now[0] += 1.0
    source.read()
    source.read()

    assert len(sleeps) == 1
    assert abs(sleeps[0] - 1 / 60) < 1e-9
