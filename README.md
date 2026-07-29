# BirdCam - Adaptive Zoom Bird Feeder Livestream Tracker

TikTok livestream with adaptive zoom for bird feeders.

## Setup (Guardian452 — Windows 11 + RTX 4090)

```
git clone https://github.com/YEE/birdcam-tracker.git
cd birdcam-tracker
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_model.py
copy config.example.yaml config.yaml
# Edit config.yaml with your TikTok RTMP stream key
python birdcam.py
```

## How It Works

- **No birds:** Full 4K frame view in 9:16 with black bars
- **Birds appear:** Smoothly zooms in on them
- **Multiple birds:** Adjusts zoom to fit all in frame

Black bars = placeholder for future background image.
