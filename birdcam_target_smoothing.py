"""Low-cost smoothing for detector-generated BirdCam crop targets."""

import birdcam_legacy as legacy


def _bounded_alpha(value, default):
    try:
        return legacy.clamp(float(value), 0.0, 1.0)
    except (TypeError, ValueError):
        return default


def _nonnegative_float(value, default):
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return default


class SmoothedGuidanceState(legacy.GuidanceState):
    """Filter detector jitter while accepting newly appearing birds immediately."""

    def __init__(self, tracker_config=None):
        super().__init__()
        tracker_config = tracker_config or {}
        config = tracker_config.get("target_smoothing", {}) or {}
        self._smoothing_enabled = bool(config.get("enabled", True))
        self._center_alpha = _bounded_alpha(config.get("center_alpha", 0.30), 0.30)
        self._size_alpha = _bounded_alpha(config.get("size_alpha", 0.12), 0.12)
        self._pan_dead_zone = _nonnegative_float(
            config.get("pan_dead_zone_pixels", 30),
            30.0,
        )
        self._zoom_dead_zone = _nonnegative_float(
            config.get("zoom_dead_zone_fraction", 0.04),
            0.04,
        )
        self._immediate_on_acquire = bool(config.get("immediate_on_acquire", True))

    @staticmethod
    def _center(crop):
        x, y, width, height = crop
        return x + width / 2, y + height / 2

    @staticmethod
    def _crop_from_center(center_x, center_y, width, height):
        return (
            int(round(center_x - width / 2)),
            int(round(center_y - height / 2)),
            max(1, int(round(width))),
            max(1, int(round(height))),
        )

    @staticmethod
    def _enclose(crop, required):
        """Expand crop just enough to include the newest raw detection target."""
        x1 = min(crop[0], required[0])
        y1 = min(crop[1], required[1])
        x2 = max(crop[0] + crop[2], required[0] + required[2])
        y2 = max(crop[1] + crop[3], required[1] + required[3])
        return x1, y1, x2 - x1, y2 - y1

    def _smooth_target(self, previous, target):
        previous_x, previous_y = self._center(previous)
        target_x, target_y = self._center(target)
        previous_w, previous_h = previous[2], previous[3]
        target_w, target_h = target[2], target[3]

        if max(abs(target_x - previous_x), abs(target_y - previous_y)) <= self._pan_dead_zone:
            center_x, center_y = previous_x, previous_y
        else:
            center_x = previous_x + self._center_alpha * (target_x - previous_x)
            center_y = previous_y + self._center_alpha * (target_y - previous_y)

        width_change = abs(target_w - previous_w) / max(previous_w, 1)
        height_change = abs(target_h - previous_h) / max(previous_h, 1)
        if max(width_change, height_change) <= self._zoom_dead_zone:
            width, height = previous_w, previous_h
        else:
            width = previous_w + self._size_alpha * (target_w - previous_w)
            height = previous_h + self._size_alpha * (target_h - previous_h)

        smoothed = self._crop_from_center(center_x, center_y, width, height)
        return self._enclose(smoothed, target)

    def publish(self, target, bird_count, observed_at, published_at=None):
        with self._lock:
            previous_count = self._bird_count
            previous_target = self._target
            self._bird_count = bird_count
            self._updated_at = observed_at
            if not bird_count or target is None:
                return

            first_target = previous_target is None
            newly_added_bird = bird_count > previous_count
            if (
                not self._smoothing_enabled
                or first_target
                or (self._immediate_on_acquire and newly_added_bird)
            ):
                self._target = target
            else:
                self._target = self._smooth_target(previous_target, target)
            self._last_birds_at = observed_at
