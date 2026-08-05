"""Letterbox presentation helpers for BirdCam output frames."""

import birdcam_legacy as legacy

LETTERBOX_GRAY = 16


def apply_dark_gray_letterbox(frame, crop, gray=LETTERBOX_GRAY):
    """Replace only the known letterbox bars with a very dark gray fill."""
    crop_width, crop_height = crop[2], crop[3]
    if legacy._aspect_ratios_match(
        crop_width,
        crop_height,
        legacy.OUT_W,
        legacy.OUT_H,
    ):
        return frame

    scale = min(
        legacy.OUT_W / crop_width,
        legacy.OUT_H / crop_height,
    )
    render_width = min(
        legacy.OUT_W,
        max(1, int(round(crop_width * scale))),
    )
    render_height = min(
        legacy.OUT_H,
        max(1, int(round(crop_height * scale))),
    )
    gray = max(0, min(255, int(gray)))
    offset_x = (legacy.OUT_W - render_width) // 2
    offset_y = (legacy.OUT_H - render_height) // 2

    if offset_y:
        frame[:offset_y, :] = gray
        frame[offset_y + render_height:, :] = gray
    if offset_x:
        frame[:, :offset_x] = gray
        frame[:, offset_x + render_width:] = gray
    return frame
