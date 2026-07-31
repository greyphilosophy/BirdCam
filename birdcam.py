"""Compatibility entry point for the optimized BirdCam pipeline."""

import birdcam_framing as _framing
import birdcam_optimized as _optimized

# Patch only the optimized runtime. Public geometry helpers retain their
# historical behavior for callers and focused unit tests.
_optimized.compute_bird_crop = _framing.compute_bird_crop
_optimized.advance_crop = _framing.advance_crop

from birdcam_optimized import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    main()
