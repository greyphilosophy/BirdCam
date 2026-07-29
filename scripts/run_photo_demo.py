"""Run the complete BirdCam pipeline using a folder of still photographs."""

import argparse
import copy
import sys
from pathlib import Path

# Support the documented direct invocation: python scripts/run_photo_demo.py ...
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import yaml

from birdcam import BirdCam
from photo_source import PhotoSequenceCapture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photos", help='Glob such as "test_photos/*.jpg"')
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seconds-per-photo", type=float, default=2.0)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Send the demo to the configured RTMP endpoint. Disabled by default.",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config = copy.deepcopy(config)
    if not args.stream:
        config.setdefault("stream", {})["rtmp_url"] = ""

    fps = float(config.get("stream", {}).get("fps", 60))
    app = BirdCam(config)
    app.open_camera = lambda: PhotoSequenceCapture(
        args.photos,
        fps=fps,
        seconds_per_photo=args.seconds_per_photo,
        loop=True,
        realtime=True,
    )
    app.run()


if __name__ == "__main__":
    main()
