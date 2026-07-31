# BirdCam — Adaptive Zoom Bird Feeder Livestream Tracker

A smooth 1080×1920 at 60 fps bird-feeder livestream guided by occasional AI detections from a 4K60 camera.

**Target birds:** Steller's Jays (COCO class 16: "bird")

## How It Works

BirdCam deliberately separates the audience-facing video path from the slower AI guidance path:

```text
ELP camera at 4K60
        ↓
latest-frame slot (old frames are discarded)
        ├── guidance worker samples newest frame at up to 5 fps
        │       ↓
        │   latest target crop
        │
        └── 60 fps renderer applies current crop
                ↓
            one downscale to 1080×1920
                ↓
            FFmpeg / TikTok RTMP
```

The renderer never waits for YOLO. It keeps using the last known target while guidance processes a newer frame. Crop position and size advance every output frame using time-based speed limits, so a slow or irregular detector cannot create camera jumps or sudden zooms.

- **No birds:** Uses the widest centered portrait crop.
- **Birds appear:** Guidance updates the target crop; the renderer approaches it smoothly.
- **Multiple birds:** The target expands to include them when they fit.
- **Birds leave:** The previous target is held briefly, then the renderer smoothly returns to the overview.
- **Guidance falls behind:** Stale input frames are discarded; only the newest frame is analyzed.

## Setup (Guardian452 — Windows 11 + RTX 4090)

### 1. Clone and Install

```cmd
git clone https://github.com/greyphilosophy/BirdCam.git
cd BirdCam
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download YOLOv8 Model

```cmd
python scripts/download_model.py
```

### 3. Configure

```cmd
copy config.example.yaml config.yaml
```

Edit `config.yaml` with your settings:

- `camera.device`: Camera device index, usually 0 or 1
- `detector.max_fps`: Maximum AI guidance rate; 5 fps is the default
- `tracker.max_zoom_fraction_per_second`: Maximum crop-size change per second
- `tracker.max_pan_fraction_per_second`: Maximum crop-center movement per second
- `stream.rtmp_url`: TikTok RTMP URL and stream key

### 4. Run

```cmd
python birdcam.py
```

## Camera diagnostics

Windows camera drivers often report the requested frame rate even when they deliver fewer frames or silently substitute another pixel format. The diagnostic utility probes likely modes using both DirectShow and Media Foundation, then measures completed frame reads with a monotonic clock.

Run the short 4K/MJPEG-focused probe first:

```cmd
python scripts\camera_diagnostic.py --quick
```

Run the complete probe matrix:

```cmd
python scripts\camera_diagnostic.py
```

Probe one exact request for a longer interval:

```cmd
python scripts\camera_diagnostic.py --backend dshow --duration 10 --mode MJPG:3840x2160:60
```

Each result shows:

- requested pixel format, resolution, and frame rate
- negotiated values reported by the driver
- actual delivered FPS measured from successful reads
- median and 95th-percentile frame intervals
- failed reads and silent format/resolution fallbacks

OpenCV does not provide reliable UVC capability enumeration on Windows, so this script probes a practical matrix rather than claiming to list every firmware-supported mode.

## Hardware

- **Camera:** ELP High Speed 4K 60FPS USB Camera (IMX678 sensor)
- **GPU:** NVIDIA RTX 4090 for YOLOv8 and NVENC
- **Output:** 1080×1920 at 60 fps for TikTok

## Performance Philosophy

BirdCam does not run AI over every 4K input frame. The 4K stream is used as a continuously refreshed source image. Each output tick crops the newest available source frame and downsizes only that region to 1080p. Guidance can run at a few detections per second without reducing output frame rate.

The debug FPS log measures the audience-facing output path. It also reports the age of the most recent guidance result so capture/render performance and AI responsiveness can be evaluated separately.
