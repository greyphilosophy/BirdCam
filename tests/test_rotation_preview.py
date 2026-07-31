import cv2
import numpy as np
import pytest

from birdcam import fit_preview, rotate_frame


def test_rotate_frame_90_degrees_clockwise():
    frame = np.array(
        [
            [[1], [2], [3]],
            [[4], [5], [6]],
        ],
        dtype=np.uint8,
    )

    rotated = rotate_frame(frame, 90)

    assert rotated.shape == (3, 2, 1)
    assert rotated[:, :, 0].tolist() == [[4, 1], [5, 2], [6, 3]]


def test_rotate_frame_accepts_full_turns():
    frame = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)

    assert rotate_frame(frame, 360) is frame
    assert np.array_equal(rotate_frame(frame, 450), cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE))


def test_rotate_frame_rejects_unsupported_angles():
    frame = np.zeros((2, 3, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="0, 90, 180, or 270"):
        rotate_frame(frame, 45)


def test_fit_preview_preserves_complete_portrait_frame():
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    frame[0, :, :] = 10
    frame[-1, :, :] = 20

    preview = fit_preview(frame, max_width=960, max_height=900)

    assert preview.shape == (900, 506, 3)
    assert np.all(preview[0] == 10)
    assert np.all(preview[-1] == 20)


def test_fit_preview_does_not_enlarge_small_frames():
    frame = np.zeros((200, 100, 3), dtype=np.uint8)

    assert fit_preview(frame, max_width=960, max_height=900) is frame
