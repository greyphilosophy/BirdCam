"""Compatibility entry point for the optimized BirdCam pipeline."""

import birdcam_framing as _framing
import birdcam_legacy as _legacy
import birdcam_optimized as _optimized

# Preserve the historical public geometry API while routing the optimized
# production classes through the stricter framing rules.
_public_compute_bird_crop = _legacy.compute_bird_crop
_public_advance_crop = _legacy.advance_crop

_legacy.compute_bird_crop = _framing.compute_bird_crop
_legacy.advance_crop = _framing.advance_crop
_optimized.compute_bird_crop = _framing.compute_bird_crop
_optimized.advance_crop = _framing.advance_crop

from birdcam_optimized import *  # noqa: F401,F403,E402

compute_bird_crop = _public_compute_bird_crop
advance_crop = _public_advance_crop


if __name__ == "__main__":
    main()
