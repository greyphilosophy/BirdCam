# BirdCam — Adaptive Zoom Bird Feeder Livestream Tracker

A smooth 1080×1920 at 60 fps bird-feeder livestream guided by occasional AI detections from a 4K60 camera.

**Target birds:** Steller's Jays (COCO class 14: "bird")

## How It Works

BirdCam deliberately separates the audience-facing video path from the slower AI guidance path:

```text
ELP camera at 4K60
        ↓
latest-frame slot (old frames are discarded)
        ├── guidance worker samples newest frame at up to 5 fps
        │       ↓
        │   dead-zone and target smoothing
        │       ↓
        │   latest target crop
        │
        └── 60 fps renderer applies current crop
                ↓
            one downscale to 1080×1920
                ├── FFmpeg / TikTok RTMP
                └── optional direct virtual camera
```

The renderer never waits for YOLO. It keeps using the last known target while guidance processes a newer frame. Crop position and dimensions advance every output frame using time-based speed limits, so a slow or irregular detector cannot create camera jumps or sudden zooms.

- **No birds yet:** Shows the complete 16:9 camera frame, centered in the vertical stream with black bars above and below.
- **Birds appear:** The first target is accepted immediately; the renderer smoothly approaches it.
- **Birds remain:** Small detector-box fluctuations are ignored and safe contraction/recentering is smoothed.
- **Multiple birds:** A newly increased bird count is accepted immediately, and meaningful target expansion always remains large enough to contain the latest detected group.
- **Birds leave:** The previous target is held briefly, then the view opens to the portrait overview and finally the full letterboxed idle view.
- **Guidance falls behind:** Stale input frames are discarded; only the newest frame is analyzed.

The transition between the wide idle view and portrait tracking is not a hard cut. BirdCam changes the crop width, height, and center using the same configured speed limits used for bird tracking. The black bars therefore grow or shrink gradually as the horizontal field of view changes. Transitions between two portrait tracking crops remain locked to the portrait aspect ratio, so ordinary bird-following zooms do not introduce black bars.

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

`requirements.txt` pins PyTorch 2.11 and torchvision 0.26. On Windows it explicitly selects the CUDA 12.8 wheels from PyTorch's package index, preventing pip from silently installing the CPU-only build. Non-Windows environments use the matching standard wheels.

Verify the environment before launching BirdCam:

```powershell
python -c "import sys, numpy, cv2, torch; print('Python:', sys.executable); print('NumPy:', numpy.__version__); print('OpenCV:', cv2.__version__); print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

On the production RTX 4090 computer, the result must show:

```text
Python: ...\BirdCam\venv\Scripts\python.exe
Torch: 2.11.0+cu128
CUDA: True
GPU: NVIDIA GeForce RTX 4090
```

If an existing environment reports a `+cpu` PyTorch build, reinstall from the updated requirements file:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --no-cache-dir --force-reinstall -r requirements.txt
```

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
- `detector.max_fps`: Maximum AI guidance rate; 5 fps is the default
- `tracker.max_zoom_fraction_per_second`: Maximum crop-size change per second
- `tracker.max_pan_fraction_per_second`: Maximum crop-center movement per second
- `tracker.target_smoothing`: Detector-target dead zones and low-frequency smoothing
- `idle_view.enabled`: Enable or disable the wide letterboxed waiting view
- `idle_view.delay_seconds`: Time since the last bird before targeting the full-frame idle view
- `debug.preview_rotation`: Optional additional rotation applied only to the local preview
- `debug.preview_width` / `preview_height`: Size of the resizable desktop preview window
- `stream.rtmp_url`: TikTok RTMP URL and stream key
- `virtual_camera.enabled`: Publish the finished BirdCam frame as a webcam device

### 4. Run

```powershell
python birdcam.py
```

For the current sideways camera mounting, use `camera.rotation_degrees: 90` and
`debug.preview_rotation: none`. Source rotation already makes detection, the
1080×1920 stream, and the portrait debug preview upright. Applying another
clockwise preview rotation would rotate the already-corrected image a second
time. Preview rotation remains available for unusual display arrangements, but
it is independent of camera/source rotation.

## Tracking smoothing

BirdCam filters guidance targets only when a new detector result arrives, normally at up to `detector.max_fps` (5 fps by default). The 60 fps capture, render, RTMP, and virtual-camera paths are unchanged.

The default configuration is:

```yaml
tracker:
  target_smoothing:
    enabled: true
    center_alpha: 0.30
    size_alpha: 0.12
    pan_dead_zone_pixels: 30
    zoom_dead_zone_fraction: 0.04
```

The first detection, a genuine reacquisition after the hold interval, and an increased bird count are always accepted immediately. Brief empty detector samples inside `tracker.hold_seconds` remain continuous tracking instead of resetting the filter. Small center and size fluctuations are held inside the dead zones. Meaningful expansion is never reduced below the newest raw detection target, while safe contraction and recentering use the configured smoothing values. The existing per-frame pan and zoom limits still control what viewers see, so immediate guidance does not become an abrupt video cut.

Because this filter performs only a fixed number of scalar operations per guidance update, it should not materially affect output frame rate.

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

For virtual-camera-only operation, leave `stream.rtmp_url` empty or remove it from your local configuration. The virtual camera will still run at `stream.fps`, which defaults to 60.

If startup reports that no virtual camera is available, confirm that the OBS Virtual Camera driver is installed and that another application is not exclusively holding the device.

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

BirdCam does not run AI over every 4K input frame. The 4K stream is used as a continuously refreshed source image. Each output tick crops the newest available source frame and downsizes only that region to the vertical output. Guidance can run at a few detections per second without reducing output frame rate.

The renderer caches the most recent finished frame when neither the source frame nor crop has changed. Debug logs report audience-facing output FPS, measured capture FPS, resize cost, current view mode, and guidance age separately.
