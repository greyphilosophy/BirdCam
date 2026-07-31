"""Apply the rotated, size-limited debug preview update to this branch.

This script is executed once by a temporary branch workflow, then removes both
itself and that workflow before the resulting application commit is pushed.
"""

from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} anchor, found {count}")
    return text.replace(old, new, 1)


root = Path(__file__).resolve().parents[1]

birdcam_path = root / "birdcam.py"
birdcam = birdcam_path.read_text(encoding="utf-8")

birdcam = replace_once(
    birdcam,
    "MAX_MOTION_STEP_SECONDS = 0.1\nCrop = tuple[int, int, int, int]\n",
    "MAX_MOTION_STEP_SECONDS = 0.1\n"
    "DEBUG_WINDOW_NAME = \"BirdCam\"\n"
    "DEFAULT_PREVIEW_WIDTH = 1280\n"
    "DEFAULT_PREVIEW_HEIGHT = 720\n"
    "Crop = tuple[int, int, int, int]\n",
    "preview constants",
)

birdcam = replace_once(
    birdcam,
    "    return choices[normalized]\n\n\n@dataclass(frozen=True)\nclass FrameSnapshot:\n",
    '''    return choices[normalized]\n\n\ndef prepare_debug_preview(frame, rotation="clockwise"):\n    """Rotate the local preview without changing the stream output frame."""\n    normalized = str(rotation or "none").strip().lower()\n    if normalized in {"none", "off", "0"}:\n        return frame\n\n    rotations = {\n        "clockwise": cv2.ROTATE_90_CLOCKWISE,\n        "cw": cv2.ROTATE_90_CLOCKWISE,\n        "90": cv2.ROTATE_90_CLOCKWISE,\n        "counterclockwise": cv2.ROTATE_90_COUNTERCLOCKWISE,\n        "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,\n        "-90": cv2.ROTATE_90_COUNTERCLOCKWISE,\n        "180": cv2.ROTATE_180,\n    }\n    if normalized not in rotations:\n        raise ValueError(\n            f"Unsupported preview rotation: {rotation!r}; "\n            "use clockwise, counterclockwise, 180, or none"\n        )\n    return cv2.rotate(frame, rotations[normalized])\n\n\ndef configure_debug_window(debug, window_name=DEBUG_WINDOW_NAME):\n    """Create a resizable, aspect-preserving preview that fits on screen."""\n    width = max(1, int(debug.get("preview_width", DEFAULT_PREVIEW_WIDTH)))\n    height = max(1, int(debug.get("preview_height", DEFAULT_PREVIEW_HEIGHT)))\n    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)\n    cv2.resizeWindow(window_name, width, height)\n    return width, height\n\n\n@dataclass(frozen=True)\nclass FrameSnapshot:\n''',
    "preview helpers",
)

birdcam = replace_once(
    birdcam,
    '''        debug = self.config.get("debug", {})\n        capture = CaptureWorker(self.open_camera, self.latest_frame, self.stop_event)\n''',
    '''        debug = self.config.get("debug", {})\n        preview_enabled = bool(debug.get("window", False))\n        preview_rotation = debug.get("preview_rotation", "clockwise")\n        if preview_enabled:\n            preview_width, preview_height = configure_debug_window(debug)\n            logger.info(\n                "Debug preview: %dx%d window, rotation=%s (stream remains 1080x1920)",\n                preview_width,\n                preview_height,\n                preview_rotation,\n            )\n        capture = CaptureWorker(self.open_camera, self.latest_frame, self.stop_event)\n''',
    "debug setup",
)

birdcam = replace_once(
    birdcam,
    '''                if debug.get("window", False):\n                    cv2.imshow("BirdCam", output)\n                    if cv2.waitKey(1) & 0xFF == ord("q"):\n                        self.stop_event.set()\n''',
    '''                if preview_enabled:\n                    preview = prepare_debug_preview(output, preview_rotation)\n                    cv2.imshow(DEBUG_WINDOW_NAME, preview)\n                    if cv2.waitKey(1) & 0xFF == ord("q"):\n                        self.stop_event.set()\n''',
    "debug rendering",
)

birdcam_path.write_text(birdcam, encoding="utf-8")

config_path = root / "config.example.yaml"
config = config_path.read_text(encoding="utf-8")
config = replace_once(
    config,
    '''debug:\n  window: true\n  log_fps: true\n''',
    '''debug:\n  window: true\n  log_fps: true\n  preview_rotation: clockwise  # Rotate only the local preview; the vertical stream is unchanged\n  preview_width: 1280          # Keep the preview inside a normal desktop window\n  preview_height: 720\n''',
    "debug configuration",
)
config_path.write_text(config, encoding="utf-8")

readme_path = root / "README.md"
readme = readme_path.read_text(encoding="utf-8")
readme = replace_once(
    readme,
    '''- `idle_view.delay_seconds`: Time since the last bird before targeting the full-frame idle view\n- `stream.rtmp_url`: TikTok RTMP URL and stream key\n''',
    '''- `idle_view.delay_seconds`: Time since the last bird before targeting the full-frame idle view\n- `debug.preview_rotation`: Rotation for the local preview; defaults to `clockwise`\n- `debug.preview_width` / `preview_height`: Size of the resizable desktop preview window\n- `stream.rtmp_url`: TikTok RTMP URL and stream key\n''',
    "README configuration list",
)
readme = replace_once(
    readme,
    '''```cmd\npython birdcam.py\n```\n\n## Camera diagnostics\n''',
    '''```cmd\npython birdcam.py\n```\n\nThe local debug preview is rotated 90 degrees clockwise and displayed in a\n1280×720 resizable window by default. This keeps the complete 1080×1920 canvas\nvisible and its letterboxed content centered on an ordinary landscape monitor.\nPreview rotation does not rotate or otherwise alter the livestream sent to\nFFmpeg. Set `debug.preview_rotation: none` to disable it.\n\n## Camera diagnostics\n''',
    "README preview note",
)
readme_path.write_text(readme, encoding="utf-8")

# The workflow and this script are implementation scaffolding only. Remove both
# before committing so the pull request contains only the application change.
(root / ".github" / "workflows" / "apply-preview-patch.yml").unlink()
Path(__file__).unlink()
