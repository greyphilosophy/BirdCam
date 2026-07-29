# BirdCam — Adaptive Zoom Bird Feeder Livestream Tracker

Variable-zoom 9:16 crop for bird feeder livestreaming on TikTok.

**Target birds:** Steller's Jays (COCO class 16: "bird")

## How It Works

- **No birds:** Shows full 4K frame scaled to 9:16 with black bars
- **Birds appear:** Smoothly zooms in on them
- **Multiple birds:** Expands crop to include all birds
- **Birds leave:** Holds position briefly, then zooms back out

Black bars serve as placeholders for future background imagery.

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
- `camera.device`: Camera device index (usually 0 or 1)
- `stream.rtmp_url`: Your TikTok RTMP URL + stream key

### 4. Run

```cmd
python birdcam.py
```

## Hardware

- **Camera:** ELP High Speed 4K 60FPS USB Camera (IMX678 sensor)
- **GPU:** NVIDIA RTX 4090 (YOLOv8 runs via CUDA)
- **Output:** 1080×1920 (9:16) for TikTok

## Config

See `config.example.yaml` for:
- Camera device index
- Bird detection confidence threshold
- Zoom hold duration
- Streaming URL and bitrate
- Debug window toggle
