"""Framing rules layered onto the optimized BirdCam pipeline."""

import birdcam_legacy as legacy

# Preserve the original motion primitive before birdcam.py patches legacy
# references to route the optimized runtime through this module.
_LEGACY_ADVANCE_CROP = legacy.advance_crop


def ensure_minimum_output_crop(crop, frame_w, frame_h):
    """Expand a crop so source pixels are never upscaled into the output."""
    x, y, width, height = crop
    if width >= legacy.OUT_W and height >= legacy.OUT_H:
        return legacy.clamp_rect(x, y, width, height, frame_w, frame_h)

    center_x = x + width / 2
    center_y = y + height / 2
    if legacy._aspect_ratios_match(width, height, legacy.OUT_W, legacy.OUT_H):
        width = max(width, legacy.OUT_W)
        height = max(height, legacy.OUT_H)
        if width / height > legacy.OUT_ASPECT:
            height = int(round(width / legacy.OUT_ASPECT))
        else:
            width = int(round(height * legacy.OUT_ASPECT))
    else:
        width = max(width, legacy.OUT_W)
        height = max(height, legacy.OUT_H)

    width = min(width, frame_w)
    height = min(height, frame_h)
    return legacy.clamp_rect(
        int(round(center_x - width / 2)),
        int(round(center_y - height / 2)),
        width,
        height,
        frame_w,
        frame_h,
    )


def _minimum_enclosing_group_crop(x1, y1, x2, y2, frame_w, frame_h):
    """Return the smallest available crop containing a padded bird group."""
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    crop_w = min(frame_w, max(legacy.OUT_W, int(round(x2 - x1))))
    crop_h = min(frame_h, max(legacy.OUT_H, int(round(y2 - y1))))
    return legacy.clamp_rect(
        int(round(center_x - crop_w / 2)),
        int(round(center_y - crop_h / 2)),
        crop_w,
        crop_h,
        frame_w,
        frame_h,
    )


def compute_bird_crop(birds, frame_w, frame_h, padding=200):
    """Frame the union of every bird without zooming below output resolution."""
    birds = list(birds)
    if not birds:
        return None

    x1 = min(box[0] for box in birds) - padding
    y1 = min(box[1] for box in birds) - padding
    x2 = max(box[2] for box in birds) + padding
    y2 = max(box[3] for box in birds) + padding
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    needed_w = max(legacy.OUT_W, x2 - x1)
    needed_h = max(legacy.OUT_H, y2 - y1)

    # Expanding a portrait crop to contain a horizontally spread group adds
    # unrelated vertical scenery. For multiple birds, preserve the complete
    # padded group with the smallest available wider crop instead; the renderer
    # can letterbox it into the portrait output.
    group_is_wider_than_portrait = needed_w / needed_h > legacy.OUT_ASPECT
    if len(birds) >= 2 and group_is_wider_than_portrait:
        return _minimum_enclosing_group_crop(x1, y1, x2, y2, frame_w, frame_h)

    if needed_w / needed_h > legacy.OUT_ASPECT:
        crop_w = needed_w
        crop_h = crop_w / legacy.OUT_ASPECT
    else:
        crop_h = needed_h
        crop_w = crop_h * legacy.OUT_ASPECT

    maximum = legacy.overview_crop(frame_w, frame_h)
    if crop_w > maximum[2] or crop_h > maximum[3]:
        return legacy.full_frame_crop(frame_w, frame_h)

    crop_w = max(legacy.OUT_W, int(round(crop_w)))
    crop_h = max(legacy.OUT_H, int(round(crop_w / legacy.OUT_ASPECT)))
    return legacy.clamp_rect(
        int(round(center_x - crop_w / 2)),
        int(round(center_y - crop_h / 2)),
        crop_w,
        crop_h,
        frame_w,
        frame_h,
    )


def advance_crop(
    current,
    target,
    elapsed,
    frame_w,
    frame_h,
    max_zoom_fraction_per_second,
    max_pan_fraction_per_second,
):
    """Advance while enforcing the no-upscale crop floor on every frame."""
    current = ensure_minimum_output_crop(current, frame_w, frame_h)
    target = ensure_minimum_output_crop(target, frame_w, frame_h)
    advanced = _LEGACY_ADVANCE_CROP(
        current,
        target,
        elapsed,
        frame_w,
        frame_h,
        max_zoom_fraction_per_second,
        max_pan_fraction_per_second,
    )
    return ensure_minimum_output_crop(advanced, frame_w, frame_h)
