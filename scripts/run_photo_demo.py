"""Run the complete BirdCam pipeline using a folder of still photographs."""

import argparse

import yaml

from birdcam import BirdCam
from photo_source import PhotoSequenceCapture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photos", help='Glob such as "test_photos/*.jpg"')
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seconds-per-photo", type=float, default=2.0)
    parser.add_argument("--no-loop", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    fps = float(config.get("stream", {}).get("fps", 60))
    app = BirdCam(config)
    app.open_camera = lambda: PhotoSequenceCapture(
        args.photos,
        fps=fps,
        seconds_per_photo=args.seconds_per_photo,
        loop=not args.no_loop,
        realtime=True,
    )
    app.run()


if __name__ == "__main__":
    main()
