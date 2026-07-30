# BirdCam test photos

Place the original full-resolution feeder photographs in this folder for local camera-emulation and framing tests.

Recommended naming uses a numeric prefix so playback order is deterministic, for example:

```text
01-wide-feeder.jpg
02-two-birds.jpg
03-left-bird-close.jpg
04-right-bird-close.jpg
05-railing-bird.jpg
```

Run the local-only photo demo from the repository root:

```powershell
python scripts/run_photo_demo.py "test_photos/*.jpg" --seconds-per-photo 2
```

The demo does not stream to RTMP unless `--stream` is explicitly supplied.

Use original-resolution images when possible. The emulator center-crops them to the configured camera aspect ratio and prepares frames at the configured camera dimensions, normally 3840×2160, so the crop, memory, and resize path resembles the intended camera feed.
