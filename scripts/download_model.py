#!/usr/bin/env python3
"""Download the YOLOv8 model file for BirdCam."""

import os
import urllib.request
from pathlib import Path

MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
OUTPUT_DIR = Path("models")
OUTPUT_FILE = OUTPUT_DIR / "yolov8n.pt"


def download():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
        print(f"Model already exists: {OUTPUT_FILE} ({size_mb:.1f} MB)")
        return

    print(f"Downloading YOLOv8 model to {OUTPUT_FILE}...")
    urllib.request.urlretrieve(MODEL_URL, OUTPUT_FILE)
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Downloaded: {OUTPUT_FILE} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    download()
