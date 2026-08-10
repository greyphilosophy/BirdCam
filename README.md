# BirdCam — Adaptive Zoom Bird Feeder Livestream Tracker

A smooth 1080×1920 at 60 fps bird-feeder livestream guided by occasional AI detections from a 4K60 camera.

**Target birds:** Steller's Jays (COCO class 14: "bird")

## How It Works

BirdCam deliberately separates the audience-facing video path from the slower AI guidance path:

```text
ELP camera at 4K60
        ↓
latest-frame slot (old frames are discarded)
        ├── guidance worker samples newest frame at up to 10 fps
        │       ↓
        │   suspicious-change confirmation
        │       ↓
        │   dead-zone and target smoothing
        │       ↓
        │   latest target crop
        │
        └── 60 fps renderer applies current crop
                ↓
            one downscale to 1080×1920
                ├── FFmpeg / RTMP
                └── optional direct virtual camera
```

The renderer never waits for YOLO. It keeps using the last known target while guidance processes a newer frame. Crop position and dimensions advance every output frame using time-based speed limits, so a slow or irregular detector cannot create camera jumps or sudden zooms.

- **No birds yet:** Shows the complete 16:9 camera frame, centered in the vertical stream with dark-gray bars above and below.
- **Birds appear:** A new target must be seen consistently before the camera begins reframing.
- **Birds remain:** Small detector-box fluctuations are ignored and safe contraction/recentering is smoothed.
- **Large target changes:** Large pan/zoom changes must agree across consecutive guidance samples before they are accepted.
- **Multiple birds:** Bird-count changes are also confirmed before they can expand or move the target.
- **Birds leave:** The previous target is held briefly, then the view opens to the portrait overview and finally the full letterboxed idle view.
- **Guidance falls behind:** Stale input frames are discarded; only the newest frame is analyzed.

The transition between the wide idle view and portrait tracking is not a hard cut. BirdCam changes the crop width, height, and center using the same configured speed limits used for bird tracking. The bars therefore grow or shrink gradually as the horizontal field of view changes. Transitions between two portrait tracking crops remain locked to the portrait aspect ratio, so ordinary bird-following zooms do not introduce bars.

## Setup (Guardian452 — Windows 11 + RTX 4090)

Use a 64-bit Python 3.10 or 3.11 installation and a current NVIDIA display driver. Do not install BirdCam into Conda's `base` environment; use a dedicated virtual environment.

### 1. Clone and install

```powershell
git clone https://github.com/greyphilosophy/BirdCam.git
cd BirdCam
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt
```

`requirements.txt` is the production Windows/RTX requirements set. It pins CUDA-enabled PyTorch wheels explicitly so pip does not silently install a CPU-only build. GitHub Actions uses `requirements-ci.txt`, which selects matching CPU-only wheels and shares the remaining pins through `requirements-common.txt`.

Verify the environment before launching BirdCam:

```powershell
python -c "import sys, numpy, cv2, torch; print('Python:', sys.executable); print('NumPy:', numpy.__version__); print('OpenCV:', cv2.__version__); print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

On the production RTX 4090 computer, CUDA must report `True` and the GPU must be `NVIDIA GeForce RTX 4090`.

If compiled NumPy or PyTorch extensions are missing, delete and recreate the entire `venv` rather than repairing packages one at a time.

### 2. Download YOLOv8 Model

```powershell
python scripts/download_model.py
```

### 3. Configure

```powershell
Copy-Item config.example.yaml config.yaml
```

Edit `config.yaml` with your settings:

- `camera.device`: Camera device index, usually 0 or 1
- `camera.backend`: Windows defaults to Media Foundation through `auto`
- `camera.rotation_degrees`: Source-frame rotation applied before detection and streaming
- `detector.max_fps`: Maximum AI guidance sampling rate; 10 fps is the example default
- `tracker.max_zoom_fraction_per_second`: Maximum crop-size change per second
- `tracker.max_pan_fraction_per_second`: Maximum crop-center movement per second
- `tracker.target_confirmation`: Consecutive-sample confirmation for suspicious target changes
- `tracker.target_smoothing`: Detector-target dead zones and low-frequency smoothing
- `idle_view.enabled`: Enable or disable the wide letterboxed waiting view
- `idle_view.delay_seconds`: Time since the last bird before targeting the full-frame idle view
- `debug.preview_rotation`: Optional additional rotation applied only to the local preview
- `debug.preview_width` / `preview_height`: Size of the resizable desktop preview window
- `stream.rtmp_url`: Optional RTMP destination
- `virtual_camera.enabled`: Publish the finished BirdCam frame as a webcam device

### 4. Run

```powershell
python birdcam.py
```

For a sideways camera mounting, use `camera.rotation_degrees: 90` and `debug.preview_rotation: none`. Source rotation already makes detection, the 1080×1920 stream, and the portrait debug preview upright.

## Tracking confirmation and smoothing

BirdCam filters guidance only when a new detector result arrives. Suspicious changes are rejected until consecutive samples agree; ordinary small movement continues through immediately. This keeps one confused YOLO frame from yanking the camera while preserving responsive tracking.

The example configuration is:

```yaml
tracker:
  target_confirmation:
    enabled: true
    required_samples: 2
    large_center_distance_pixels: 80
    large_size_change_fraction: 0.08
    agreement_center_distance_pixels: 180
    agreement_size_change_fraction: 0.20
  target_smoothing:
    enabled: true
    center_alpha: 0.30
    size_alpha: 0.12
    pan_dead_zone_pixels: 30
    zoom_dead_zone_fraction: 0.04
```

A first acquisition, a large pan or zoom change, and a bird-count change all require two agreeing samples. If the next sample contradicts the candidate, the pending change is discarded and confirmation starts over. Small continuous changes do not wait for confirmation. Once a target is accepted, the existing dead zones, smoothing, and per-frame pan/zoom velocity limits still control presentation.

At a 10 fps guidance rate, two-sample confirmation normally adds about one guidance interval of latency while preventing isolated detector mistakes from becoming visible camera motion. Confirmation and smoothing run only on guidance updates, not in the audience-facing render loop.

## Direct virtual-camera output

BirdCam can publish its completed 1080×1920 frame directly as a camera device through `pyvirtualcam`. This avoids capturing the preview window and does not require the OBS application to remain open.

On Windows, install OBS Studio once so the **OBS Virtual Camera** driver is available. OBS itself can stay closed while BirdCam is running.

Enable the output in `config.yaml`:

```yaml
virtual_camera:
  enabled: true
  backend: obs
  device: null
```

Start BirdCam normally:

```powershell
python birdcam.py
```

Then choose **OBS Virtual Camera** in the receiving application. BirdCam sends the same rendered frames used by the RTMP output, so both outputs may run at the same time.

For virtual-camera-only operation, leave `stream.rtmp_url` empty or remove it from your local configuration. The virtual camera still uses `stream.fps` as its requested cadence.

## Camera diagnostics

Windows camera drivers often report the requested frame rate even when they deliver fewer frames or silently substitute another pixel format. The diagnostic utility probes likely modes using both DirectShow and Media Foundation, then measures completed frame reads with a monotonic clock.

Run the short 4K/MJPEG-focused probe first:

```powershell
python scripts\camera_diagnostic.py --quick
```

Run the complete probe matrix:

```powershell
python scripts\camera_diagnostic.py
```

Probe one exact request for a longer interval:

```powershell
python scripts\camera_diagnostic.py --backend dshow --duration 10 --mode MJPG:3840x2160:60
```

Each result shows requested and negotiated format, actual delivered FPS, frame intervals, failed reads, and silent fallbacks.

## Hardware

- **Camera:** ELP High Speed 4K 60FPS USB Camera (IMX678 sensor)
- **GPU:** NVIDIA RTX 4090 for YOLOv8 and NVENC
- **Output:** 1080×1920 portrait video

## Performance Philosophy

BirdCam does not run AI over every 4K input frame. The 4K stream is used as a continuously refreshed source image. Each output tick crops the newest available source frame and downsizes only that region to the vertical output. Guidance can run independently without reducing the requested output cadence.

The renderer caches the most recent finished frame when neither the source frame nor crop has changed. Debug logs report audience-facing output FPS, measured capture FPS, resize cost, current view mode, and guidance age separately.
