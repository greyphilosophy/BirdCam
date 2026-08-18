"""Compatibility entry point for the optimized BirdCam pipeline."""

import sys
import time

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

    def run(self):
        detector = self.config["detector"]
        tracker = self.config["tracker"]
        try:
            _legacy.logger.info("Loading YOLO guidance model: %s", detector["model_path"])
            model = _legacy.YOLO(detector["model_path"])
            sample_period = 1.0 / max(
                0.1,
                float(detector.get("max_fps", 5.0)),
            )
            sequence = 0
            next_sample = 0.0
            while not self.stop_event.is_set():
                snapshot = self.latest_frame.wait_for_newer(
                    sequence,
                    timeout=0.25,
                )
                if snapshot is None:
                    continue
                sequence = snapshot.sequence
                if time.monotonic() < next_sample:
                    continue

                guidance_frame = _legacy.rotate_frame(
                    snapshot.frame,
                    self.rotation_degrees,
                )
                birds = _legacy.detect_birds(
                    model,
                    guidance_frame,
                    detector["conf_thresh"],
                    detector.get("imgsz", 1280),
                    detector.get("device", 0),
                )
                frame_h, frame_w = guidance_frame.shape[:2]
                if birds and hasattr(self.guidance, "set_edge_risk"):
                    self.guidance.set_edge_risk(
                        _bird_near_frame_edge(
                            birds,
                            frame_w,
                            frame_h,
                            tracker.get("padding", 0),
                        )
                    )
                self.guidance.publish(
                    _legacy.compute_bird_crop(
                        birds,
                        frame_w,
                        frame_h,
                        tracker["padding"],
                    ),
                    len(birds),
                    snapshot.captured_at,
                    published_at=time.monotonic(),
                )
                next_sample = time.monotonic() + sample_period
        except Exception as exc:
            self.error = exc
            _legacy.logger.exception("Guidance worker stopped")
            self.stop_event.set()


# BirdCam's optimized run loop resolves GuidanceWorker from its module globals at
# runtime, so swap in the edge-aware worker without duplicating the render loop.
_optimized.GuidanceWorker = EdgeAwareGuidanceWorker
_OptimizedBirdCam = BirdCam


class SmoothedGuidanceState(_SmoothedGuidanceState):
    """Smoothed framing with raw status and extra edge-loss hysteresis."""

    def __init__(self, tracker_config=None):
        super().__init__(tracker_config)
        self._observed_bird_count = 0
        self._next_edge_risk = False
        self._edge_risk = False

    def set_edge_risk(self, edge_risk):
        with self._lock:
            self._next_edge_risk = bool(edge_risk)

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
                elif self._edge_risk:
                    self._last_birds_at = observed_at
            return

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
