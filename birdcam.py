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
_OptimizedBirdCam = BirdCam


class SmoothedGuidanceState(_SmoothedGuidanceState):
    """Smoothed framing state whose status reports the latest raw detection."""

    def __init__(self, tracker_config=None):
        super().__init__(tracker_config)
        self._observed_bird_count = 0

    def publish(self, target, bird_count, observed_at, published_at=None):
        super().publish(target, bird_count, observed_at, published_at=published_at)
        with self._lock:
            self._observed_bird_count = bird_count

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
