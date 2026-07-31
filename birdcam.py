"""Compatibility entry point for the optimized BirdCam pipeline."""

import birdcam_framing as _framing
import birdcam_legacy as _legacy
import birdcam_optimized as _optimized

# The optimized classes still reference selected helpers through the legacy
# module. Patch both namespaces so production behavior matches the framing
# helpers covered by tests.
_legacy.compute_bird_crop = _framing.compute_bird_crop
_legacy.advance_crop = _framing.advance_crop
_optimized.compute_bird_crop = _framing.compute_bird_crop
_optimized.advance_crop = _framing.advance_crop

from birdcam_optimized import *  # noqa: F401,F403,E402
from birdcam_legacy import advance_crop, compute_bird_crop  # noqa: F401,E402


if __name__ == "__main__":
    main()
