"""Low-cost filtering for detector-generated BirdCam crop targets."""

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


def _positive_int(value, default):
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


class SmoothedGuidanceState(legacy.GuidanceState):
    """Reject unstable target jumps, then smooth accepted detector guidance."""

    def __init__(self, tracker_config=None):
        super().__init__()
        tracker_config = tracker_config or {}
        smoothing = tracker_config.get("target_smoothing", {}) or {}
        confirmation = tracker_config.get("target_confirmation", {}) or {}

        self._smoothing_enabled = bool(smoothing.get("enabled", True))
        self._center_alpha = _bounded_alpha(smoothing.get("center_alpha", 0.30), 0.30)
        self._size_alpha = _bounded_alpha(smoothing.get("size_alpha", 0.12), 0.12)
        self._pan_dead_zone = _nonnegative_float(
            smoothing.get("pan_dead_zone_pixels", 30),
            30.0,
        )
        self._zoom_dead_zone = _nonnegative_float(
            smoothing.get("zoom_dead_zone_fraction", 0.04),
            0.04,
        )
        self._hold_seconds = _nonnegative_float(
            tracker_config.get("hold_seconds", 1.0),
            1.0,
        )
        self._padding = _nonnegative_float(
            tracker_config.get("padding", 200),
            200.0,
        )

        self._confirmation_enabled = bool(confirmation.get("enabled", True))
        self._required_samples = _positive_int(
            confirmation.get("required_samples", 2),
            2,
        )
        self._large_center_distance = _nonnegative_float(
            confirmation.get("large_center_distance_pixels", 80),
            80.0,
        )
        self._large_size_change = _nonnegative_float(
            confirmation.get("large_size_change_fraction", 0.08),
            0.08,
        )
        self._agreement_center_distance = _nonnegative_float(
            confirmation.get("agreement_center_distance_pixels", 180),
            180.0,
        )
        self._agreement_size_change = _nonnegative_float(
            confirmation.get("agreement_size_change_fraction", 0.20),
            0.20,
        )
        self._pending_target = None
        self._pending_bird_count = 0
        self._pending_samples = 0
        self._empty_samples = 0
        self._single_bird_size_history = []
        self._stable_single_bird_size = None

    def view_for(
        self,
        frame_w,
        frame_h,
        now,
        hold_seconds,
        idle_enabled=True,
        idle_after_seconds=3.0,
    ):
        """Stop pursuing a stale bird crop once bird loss is confirmed."""
        with self._lock:
            bird_count = self._bird_count
            last_birds_at = self._last_birds_at

        if bird_count == 0 and last_birds_at is not None:
            return (
                (legacy.full_frame_crop(frame_w, frame_h), "idle")
                if idle_enabled
                else (legacy.overview_crop(frame_w, frame_h), "overview")
            )

        return super().view_for(
            frame_w,
            frame_h,
            now,
            hold_seconds,
            idle_enabled=idle_enabled,
            idle_after_seconds=idle_after_seconds,
        )

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

    @classmethod
    def _center_distance(cls, first, second):
        first_x, first_y = cls._center(first)
        second_x, second_y = cls._center(second)
        return max(abs(second_x - first_x), abs(second_y - first_y))

    @staticmethod
    def _size_change(first, second):
        width_change = abs(second[2] - first[2]) / max(first[2], 1)
        height_change = abs(second[3] - first[3]) / max(first[3], 1)
        return max(width_change, height_change)

    @classmethod
    def _ensure_contains(cls, crop, required):
        """Return the smallest crop near crop's center that contains required."""
        center_x, center_y = cls._center(crop)
        width = max(crop[2], required[2])
        height = max(crop[3], required[3])

        minimum_center_x = required[0] + required[2] - width / 2
        maximum_center_x = required[0] + width / 2
        minimum_center_y = required[1] + required[3] - height / 2
        maximum_center_y = required[1] + height / 2
        center_x = legacy.clamp(center_x, minimum_center_x, maximum_center_x)
        center_y = legacy.clamp(center_y, minimum_center_y, maximum_center_y)
        return cls._crop_from_center(center_x, center_y, width, height)

    def _reset_single_bird_size_history(self):
        self._single_bird_size_history = []
        self._stable_single_bird_size = None

    def _stabilize_single_bird_target(self, target, bird_count):
        """Filter single-bird zoom size while leaving its current center responsive."""
        if target is None or bird_count != 1:
            if bird_count and bird_count != 1:
                self._reset_single_bird_size_history()
            return target

        self._single_bird_size_history.append(target)
        if len(self._single_bird_size_history) > 5:
            self._single_bird_size_history.pop(0)

        if self._stable_single_bird_size is None:
            self._stable_single_bird_size = (target[2], target[3])

        if len(self._single_bird_size_history) >= 3:
            ordered = sorted(
                self._single_bird_size_history,
                key=lambda crop: crop[2] * crop[3],
            )
            median_crop = ordered[len(ordered) // 2]
            candidate_size = (median_crop[2], median_crop[3])
            stable_w, stable_h = self._stable_single_bird_size
            size_deadband = 2.0 * self._padding
            if (
                abs(candidate_size[0] - stable_w) > size_deadband
                or abs(candidate_size[1] - stable_h) > size_deadband
            ):
                self._stable_single_bird_size = candidate_size

        center_x, center_y = self._center(target)
        stable_w, stable_h = self._stable_single_bird_size
        return self._crop_from_center(center_x, center_y, stable_w, stable_h)

    def _smooth_target(self, previous, target):
        previous_x, previous_y = self._center(previous)
        target_x, target_y = self._center(target)
        previous_w, previous_h = previous[2], previous[3]
        target_w, target_h = target[2], target[3]

        pan_is_jitter = self._center_distance(previous, target) <= self._pan_dead_zone
        size_is_jitter = self._size_change(previous, target) <= self._zoom_dead_zone

        if pan_is_jitter and size_is_jitter:
            return previous

        if pan_is_jitter:
            center_x, center_y = previous_x, previous_y
        else:
            center_x = previous_x + self._center_alpha * (target_x - previous_x)
            center_y = previous_y + self._center_alpha * (target_y - previous_y)

        if size_is_jitter:
            width, height = previous_w, previous_h
        else:
            width = previous_w + self._size_alpha * (target_w - previous_w)
            height = previous_h + self._size_alpha * (target_h - previous_h)

        smoothed = self._crop_from_center(center_x, center_y, width, height)
        return self._ensure_contains(smoothed, target)

    def _clear_pending(self):
        self._pending_target = None
        self._pending_bird_count = 0
        self._pending_samples = 0

    def _requires_confirmation(self, target, bird_count):
        if not self._confirmation_enabled or self._required_samples <= 1:
            return False
        if self._target is None:
            return True
        if bird_count != self._bird_count:
            return True
        return (
            self._center_distance(self._target, target) > self._large_center_distance
            or self._size_change(self._target, target) > self._large_size_change
        )

    def _candidate_agrees(self, target, bird_count):
        return (
            self._pending_target is not None
            and bird_count == self._pending_bird_count
            and self._center_distance(self._pending_target, target)
            <= self._agreement_center_distance
            and self._size_change(self._pending_target, target)
            <= self._agreement_size_change
        )

    def _confirm_or_hold(self, target, bird_count):
        """Return True only after enough consecutive suspicious samples agree."""
        if not self._requires_confirmation(target, bird_count):
            self._clear_pending()
            return True

        if self._candidate_agrees(target, bird_count):
            # Keep the first candidate as the anchor so a longer confirmation
            # sequence cannot drift across the frame one small step at a time.
            self._pending_samples += 1
        else:
            self._pending_target = target
            self._pending_bird_count = bird_count
            self._pending_samples = 1

        if self._pending_samples < self._required_samples:
            return False

        self._clear_pending()
        return True

    def publish(self, target, bird_count, observed_at, published_at=None):
        with self._lock:
            previous_count = self._bird_count
            previous_target = self._target
            last_birds_at = self._last_birds_at
            self._updated_at = observed_at

            if not bird_count or target is None:
                self._empty_samples += 1
                self._clear_pending()
                loss_confirmed = (
                    not self._confirmation_enabled
                    or self._required_samples <= 1
                    or self._empty_samples >= self._required_samples
                )
                if loss_confirmed:
                    self._bird_count = 0
                    self._reset_single_bird_size_history()
                return

            target = self._stabilize_single_bird_target(target, bird_count)
            had_empty_sample = self._empty_samples > 0
            self._empty_samples = 0
            if not self._confirm_or_hold(target, bird_count):
                # A pending replacement may extend an actively observed bird's
                # hold, but never after even one empty detector result.
                if (
                    not had_empty_sample
                    and previous_count > 0
                    and last_birds_at is not None
                    and max(0.0, observed_at - last_birds_at) <= self._hold_seconds
                ):
                    self._last_birds_at = observed_at
                return

            self._last_birds_at = observed_at
            recently_tracking = (
                last_birds_at is not None
                and max(0.0, observed_at - last_birds_at) <= self._hold_seconds
            )
            first_target = previous_target is None
            newly_added_bird = bird_count > previous_count and not (
                previous_count == 0 and recently_tracking
            )

            self._bird_count = bird_count
            if not self._smoothing_enabled or first_target or newly_added_bird:
                self._target = target
            else:
                self._target = self._smooth_target(previous_target, target)
