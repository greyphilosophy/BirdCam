"""Compatibility entry point for the optimized BirdCam pipeline."""

import birdcam_framing as _framing
import birdcam_optimized as _optimized

_optimized.compute_bird_crop = _framing.compute_bird_crop
_optimized.advance_crop = _framing.advance_crop

from birdcam_optimized import *  # noqa: F401,F403,E402
from birdcam_framing import (  # noqa: F401,E402
    advance_crop,
    compute_bird_crop,
    ensure_minimum_output_crop,
)


if __name__ == "__main__":
    main()
