"""Compatibility entry point for the optimized BirdCam pipeline."""

import sys

import yaml

import birdcam_framing as _framing
import birdcam_legacy as _legacy
import birdcam_optimized as _optimized
from birdcam_letterbox import apply_dark_gray_letterbox
from birdcam_target_smoothing import SmoothedGuidanceState as _SmoothedGuidanceState
from birdcam_virtual_camera import CompositeOutput, VirtualCameraOutput

# Preserve the historical public geometry API while routing the optimized
# production classes through the stricter framing rules.
_public_compute_bird_crop = _legacy.compute_bird_crop
_public_advance_crop = _legacy.advance_crop

_legacy.compute_bird_crop = _framing.compute_bird_crop
_legacy.advance_crop = _framing.advance_crop
_optimized.compute_bird_crop = _framing.compute_bird_crop
_optimized.advance_crop = _framing.advance_crop

# Preserve real black pixels in the camera image while recoloring only the bars
# whose exact bounds are known from the crop geometry.
_optimized_crop_and_scale_rotated = _optimized.crop_and_scale_rotated


def _crop_and_scale_with_dark_gray_letterbox(frame, crop, degrees):
    output = _optimized_crop_and_scale_rotated(frame, crop, degrees)
    return apply_dark_gray_letterbox(output, crop)


_optimized.crop_and_scale_rotated = _crop_and_scale_with_dark_gray_letterbox

from birdcam_optimized import *  # noqa: F401,F403,E402

compute_bird_crop = _public_compute_bird_crop
advance_crop = _public_advance_crop


def _bird_near_frame_edge(birds, frame_w, frame_h, padding):
    """Return whether any detected bird lies inside the padding-width edge band."""
    margin = max(0, int(padding))
    if margin <= 0:
        return False
    right_limit = max(0, frame_w - margin)
    bottom_limit = max(0, frame_h - margin)
    return any(
        x1 <= margin
        or y1 <= margin
        or x2 >= right_limit
        or y2 >= bottom_limit
        for x1, y1, x2, y2 in birds
    )


class EdgeAwareGuidanceWorker(_optimized.GuidanceWorker):
    """Publish whether the current bird observation is vulnerable to edge flicker."""

    def _on_detection(self, birds, frame_w, frame_h, tracker):
        if birds and hasattr(self.guidance, "set_edge_risk"):
            self.guidance.set_edge_risk(
                _bird_near_frame_edge(
                    birds,
                    frame_w,
                    frame_h,
                    tracker.get("padding", 0),
                )
            )


# BirdCam's optimized run loop resolves GuidanceWorker from its module globals at
# runtime, so swap in the edge-aware worker without duplicating the render loop.
_optimized.GuidanceWorker = EdgeAwareGuidanceWorker
_OptimizedBirdCam = BirdCam


class SmoothedGuidanceState(_SmoothedGuidanceState):
    """Production framing with edge-loss and single-bird zoom hysteresis."""

    def __init__(self, tracker_config=None):
        tracker_config = tracker_config or {}
        super().__init__(tracker_config)
        self._observed_bird_count = 0
        self._next_edge_risk = False
        self._edge_risk = False
        try:
            self._padding = max(0.0, float(tracker_config.get("padding", 200)))
        except (TypeError, ValueError):
            self._padding = 200.0
        self._single_bird_size_history = []
        self._stable_single_bird_size = None
        self._stable_single_bird_center = None

    def set_edge_risk(self, edge_risk):
        with self._lock:
            self._next_edge_risk = bool(edge_risk)

    def _reset_zoom_history(self):
        self._single_bird_size_history = []
        self._stable_single_bird_size = None
        self._stable_single_bird_center = None

    def _stabilize_single_bird_target(self, target, bird_count):
        if not self._smoothing_enabled or target is None or bird_count != 1:
            if bird_count and bird_count != 1:
                self._reset_zoom_history()
            return target

        self._single_bird_size_history.append(target)
        if len(self._single_bird_size_history) > 5:
            self._single_bird_size_history.pop(0)

        current_center = self._center(target)
        if self._stable_single_bird_size is None:
            self._stable_single_bird_size = (target[2], target[3])
            self._stable_single_bird_center = current_center

        stable_w, stable_h = self._stable_single_bird_size
        deadband = 2.0 * self._padding
        current_size_is_outlier = (
            abs(target[2] - stable_w) > deadband
            or abs(target[3] - stable_h) > deadband
        )

        if len(self._single_bird_size_history) >= 3:
            ordered = sorted(
                self._single_bird_size_history,
                key=lambda crop: crop[2] * crop[3],
            )
            median_crop = ordered[len(ordered) // 2]
            candidate_size = (median_crop[2], median_crop[3])
            if (
                abs(candidate_size[0] - stable_w) > deadband
                or abs(candidate_size[1] - stable_h) > deadband
            ):
                self._stable_single_bird_size = candidate_size
                stable_w, stable_h = candidate_size
                current_size_is_outlier = (
                    abs(target[2] - stable_w) > deadband
                    or abs(target[3] - stable_h) > deadband
                )

        if current_size_is_outlier and self._stable_single_bird_center is not None:
            center_x, center_y = self._stable_single_bird_center
        else:
            center_x, center_y = current_center
            self._stable_single_bird_center = current_center

        stable_w, stable_h = self._stable_single_bird_size
        return self._crop_from_center(center_x, center_y, stable_w, stable_h)

    def publish(self, target, bird_count, observed_at, published_at=None):
        if not bird_count or target is None:
            with self._lock:
                self._observed_bird_count = bird_count
                self._updated_at = observed_at
                self._empty_samples += 1
                self._clear_pending()
                if not self._confirmation_enabled or self._required_samples <= 1:
                    required_empty_samples = 1
                else:
                    required_empty_samples = self._required_samples
                    if self._edge_risk:
                        required_empty_samples *= 2
                if self._empty_samples >= required_empty_samples:
                    self._bird_count = 0
                    self._reset_zoom_history()
                elif self._edge_risk:
                    self._last_birds_at = observed_at
            return

        target = self._stabilize_single_bird_target(target, bird_count)
        super().publish(target, bird_count, observed_at, published_at=published_at)
        with self._lock:
            self._observed_bird_count = bird_count
            if self._pending_samples == 0:
                self._edge_risk = self._next_edge_risk

    def status(self):
        with self._lock:
            return self._observed_bird_count, self._updated_at


class BirdCam(_OptimizedBirdCam):
    """Optimized BirdCam with smoothed guidance and optional virtual output."""

    def __init__(self, config):
        super().__init__(config)
        self.guidance = SmoothedGuidanceState(config.get("tracker", {}))

    def start_streamer(self):
        super().start_streamer()
        stream_fps = max(
            1.0,
            float(self.config.get("stream", {}).get("fps", 60)),
        )
        virtual_camera = VirtualCameraOutput(
            self.config.get("virtual_camera", {}),
            _legacy.OUT_W,
            _legacy.OUT_H,
            stream_fps,
            _legacy.logger,
        )
        try:
            virtual_camera.start()
        except Exception:
            self.stop_event.set()
            if self.streamer is not None:
                self.streamer.stop()
                self.streamer = None
            raise
        if virtual_camera.enabled:
            self.streamer = CompositeOutput([self.streamer, virtual_camera])


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    BirdCam(config).run()


if __name__ == "__main__":
    main()
