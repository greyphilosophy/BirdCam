"""BirdCam — smooth 4K guidance feeding a 1080p60 livestream."""

import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("birdcam")

OUT_W = 1080
OUT_H = 1920
OUT_ASPECT = OUT_W / OUT_H
DEFAULT_FULL_W = 3840
DEFAULT_FULL_H = 2160
BIRD_CLASS_ID = 16
Crop = tuple[int, int, int, int]


def detect_birds(model, frame, conf_thresh=0.45, imgsz=1280, device=0):
    """Detect birds and return boxes in source-frame coordinates."""
    results = model.predict(frame, verbose=False, conf=conf_thresh, imgsz=imgsz, device=device)
    birds = []
    for result in results:
        if result.boxes is None:
            continue
        for box, class_id in zip(result.boxes.xyxy, result.boxes.cls):
            if int(class_id) == BIRD_CLASS_ID:
                birds.append(tuple(int(value) for value in box.tolist()))
    return birds


def clamp(value, lower, upper):
    return max(lower, min(value, upper))


def move_toward(current, target, maximum_delta):
    """Move one scalar toward another without overshooting."""
    if maximum_delta <= 0:
        return current
    return current + clamp(target - current, -maximum_delta, maximum_delta)


def clamp_rect(x, y, width, height, frame_w, frame_h):
    """Clamp a rectangle to the frame without changing its dimensions."""
    width = max(1, min(int(width), frame_w))
    height = max(1, min(int(height), frame_h))
    x = max(0, min(int(x), frame_w - width))
    y = max(0, min(int(y), frame_h - height))
    return x, y, width, height


def overview_crop(frame_w, frame_h):
    """Return the widest centered 9:16 crop available in the source frame."""
    height = min(frame_h, int(round(frame_w / OUT_ASPECT)))
    width = min(frame_w, int(round(height * OUT_ASPECT)))
    height = int(round(width / OUT_ASPECT))
    return clamp_rect((frame_w - width) // 2, (frame_h - height) // 2, width, height, frame_w, frame_h)


def compute_bird_crop(
    birds: Iterable[tuple[int, int, int, int]],
    frame_w: int,
    frame_h: int,
    padding: int = 200,
) -> Optional[Crop]:
    """Return the smallest valid 9:16 crop containing all birds."""
    birds = list(birds)
    if not birds:
        return None

    all_x1 = min(box[0] for box in birds)
    all_y1 = min(box[1] for box in birds)
    all_x2 = max(box[2] for box in birds)
    all_y2 = max(box[3] for box in birds)
    center_x = (all_x1 + all_x2) / 2
    center_y = (all_y1 + all_y2) / 2
    needed_w = max(1, all_x2 - all_x1 + 2 * padding)
    needed_h = max(1, all_y2 - all_y1 + 2 * padding)

    if needed_w / needed_h > OUT_ASPECT:
        crop_w = needed_w
        crop_h = crop_w / OUT_ASPECT
    else:
        crop_h = needed_h
        crop_w = crop_h * OUT_ASPECT

    maximum = overview_crop(frame_w, frame_h)
    if crop_w > maximum[2] or crop_h > maximum[3]:
        return maximum

    crop_w = max(1, int(round(crop_w)))
    crop_h = max(1, int(round(crop_w / OUT_ASPECT)))
    x = int(round(center_x - crop_w / 2))
    y = int(round(center_y - crop_h / 2))
    return clamp_rect(x, y, crop_w, crop_h, frame_w, frame_h)


def advance_crop(
    current: Crop,
    target: Crop,
    elapsed: float,
    frame_w: int,
    frame_h: int,
    max_zoom_fraction_per_second: float,
    max_pan_fraction_per_second: float,
) -> Crop:
    """Advance a crop target with time-based zoom and pan velocity limits."""
    if elapsed <= 0:
        return current

    current_x, current_y, current_w, current_h = current
    target_x, target_y, target_w, target_h = target
    max_zoom_delta = max(1.0, current_w * max_zoom_fraction_per_second * elapsed)
    width = move_toward(float(current_w), float(target_w), max_zoom_delta)
    width = clamp(width, 1.0, float(overview_crop(frame_w, frame_h)[2]))
    height = width / OUT_ASPECT
    if height > frame_h:
        height = float(frame_h)
        width = height * OUT_ASPECT

    current_cx = current_x + current_w / 2
    current_cy = current_y + current_h / 2
    target_cx = target_x + target_w / 2
    target_cy = target_y + target_h / 2
    max_pan_delta = max(frame_w, frame_h) * max_pan_fraction_per_second * elapsed
    center_x = move_toward(current_cx, target_cx, max_pan_delta)
    center_y = move_toward(current_cy, target_cy, max_pan_delta)

    width_i = max(1, int(round(width)))
    height_i = max(1, int(round(width_i / OUT_ASPECT)))
    x = int(round(center_x - width_i / 2))
    y = int(round(center_y - height_i / 2))
    return clamp_rect(x, y, width_i, height_i, frame_w, frame_h)


def crop_and_scale(frame, crop):
    """Crop the 4K source once and downscale directly to 1080x1920."""
    x, y, width, height = crop
    region = frame[y:y + height, x:x + width]
    if region.size == 0:
        raise ValueError(f"Empty crop: {crop}")
    interpolation = cv2.INTER_AREA if width >= OUT_W and height >= OUT_H else cv2.INTER_LINEAR
    return cv2.resize(region, (OUT_W, OUT_H), interpolation=interpolation)


def fourcc_to_text(value):
    value = int(value)
    return "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4))


@dataclass(frozen=True)
class FrameSnapshot:
    sequence: int
    captured_at: float
    frame: np.ndarray


class LatestFrame:
    """Thread-safe single-slot frame buffer that never accumulates stale frames."""

    def __init__(self):
        self._condition = threading.Condition()
        self._snapshot = None

    def publish(self, frame, captured_at=None):
        with self._condition:
            sequence = 1 if self._snapshot is None else self._snapshot.sequence + 1
            timestamp = time.monotonic() if captured_at is None else captured_at
            self._snapshot = FrameSnapshot(sequence, timestamp, frame)
            self._condition.notify_all()
            return sequence

    def get(self):
        with self._condition:
            return self._snapshot

    def wait_for_newer(self, sequence, timeout=None):
        with self._condition:
            self._condition.wait_for(
                lambda: self._snapshot is not None and self._snapshot.sequence > sequence,
                timeout=timeout,
            )
            if self._snapshot is None or self._snapshot.sequence <= sequence:
                return None
            return self._snapshot


class GuidanceState:
    """Atomic crop guidance shared between the detector and renderer."""

    def __init__(self):
        self._lock = threading.Lock()
        self._target = None
        self._bird_count = 0
        self._last_birds_at = None
        self._updated_at = None

    def publish(self, target, bird_count, observed_at):
        with self._lock:
            self._bird_count = bird_count
            self._updated_at = observed_at
            if bird_count:
                self._target = target
                self._last_birds_at = observed_at

    def target_for(self, frame_w, frame_h, now, hold_seconds):
        with self._lock:
            target = self._target
            last_birds_at = self._last_birds_at
        if target is None or last_birds_at is None or now - last_birds_at > hold_seconds:
            return overview_crop(frame_w, frame_h)
        return clamp_rect(*target, frame_w, frame_h)

    def status(self):
        with self._lock:
            return self._bird_count, self._updated_at


class CaptureWorker(threading.Thread):
    """Continuously capture 4K frames and replace the latest-frame slot."""

    def __init__(self, open_camera, latest_frame, stop_event):
        super().__init__(name="birdcam-capture", daemon=True)
        self.open_camera = open_camera
        self.latest_frame = latest_frame
        self.stop_event = stop_event
        self.error = None

    def run(self):
        cap = None
        failures = 0
        try:
            while not self.stop_event.is_set():
                if cap is None:
                    cap = self.open_camera()
                ok, frame = cap.read()
                if ok:
                    failures = 0
                    self.latest_frame.publish(frame)
                    continue
                failures += 1
                if failures < 10:
                    time.sleep(0.01)
                    continue
                logger.warning("Camera read failed repeatedly; reconnecting")
                cap.release()
                cap = None
                failures = 0
                self.stop_event.wait(1.0)
        except Exception as exc:
            self.error = exc
            logger.exception("Capture worker stopped")
            self.stop_event.set()
        finally:
            if cap is not None:
                cap.release()


class GuidanceWorker(threading.Thread):
    """Sample only the newest 4K frame and update crop guidance independently."""

    def __init__(self, config, latest_frame, guidance, stop_event):
        super().__init__(name="birdcam-guidance", daemon=True)
        self.config = config
        self.latest_frame = latest_frame
        self.guidance = guidance
        self.stop_event = stop_event
        self.error = None

    def run(self):
        detector = self.config["detector"]
        tracker = self.config["tracker"]
        try:
            logger.info("Loading YOLO guidance model: %s", detector["model_path"])
            model = YOLO(detector["model_path"])
            maximum_fps = max(0.1, float(detector.get("max_fps", 5.0)))
            sample_period = 1.0 / maximum_fps
            sequence = 0
            next_sample = 0.0
            while not self.stop_event.is_set():
                snapshot = self.latest_frame.wait_for_newer(sequence, timeout=0.25)
                if snapshot is None:
                    continue
                sequence = snapshot.sequence
                now = time.monotonic()
                if now < next_sample:
                    continue
                birds = detect_birds(
                    model,
                    snapshot.frame,
                    detector["conf_thresh"],
                    detector.get("imgsz", 1280),
                    detector.get("device", 0),
                )
                frame_h, frame_w = snapshot.frame.shape[:2]
                target = compute_bird_crop(birds, frame_w, frame_h, tracker["padding"])
                self.guidance.publish(target, len(birds), snapshot.captured_at)
                next_sample = time.monotonic() + sample_period
        except Exception as exc:
            self.error = exc
            logger.exception("Guidance worker stopped")
            self.stop_event.set()


class BirdCam:
    """Run independent capture, guidance, and fixed-rate rendering paths."""

    def __init__(self, config):
        self.config = config
        self.streamer = None
        self.stop_event = threading.Event()
        self.latest_frame = LatestFrame()
        self.guidance = GuidanceState()

    def open_camera(self):
        camera = self.config["camera"]
        device = camera["device"]
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(device, backend)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*camera.get("fourcc", "MJPG")))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.get("width", DEFAULT_FULL_W))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.get("height", DEFAULT_FULL_H))
        cap.set(cv2.CAP_PROP_FPS, camera.get("fps", 60))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Unable to open camera device {device}")

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = fourcc_to_text(cap.get(cv2.CAP_PROP_FOURCC))
        logger.info("Camera opened: %s @ %dx%d %.2f fps (%s)", device, actual_w, actual_h, actual_fps, actual_fourcc)
        requested = (camera.get("width", DEFAULT_FULL_W), camera.get("height", DEFAULT_FULL_H), camera.get("fps", 60))
        if (actual_w, actual_h) != requested[:2] or actual_fps + 0.5 < requested[2]:
            logger.warning("Camera did not negotiate requested mode %dx%d @ %s fps", *requested)
        if actual_fourcc.strip("\x00 ") != camera.get("fourcc", "MJPG"):
            logger.warning("Camera negotiated %s instead of %s", actual_fourcc, camera.get("fourcc", "MJPG"))
        return cap

    def start_streamer(self):
        stream = self.config.get("stream", {})
        if not stream.get("rtmp_url"):
            return
        from streamer import RTMPStreamer

        audio = self.config.get("audio", {})
        self.streamer = RTMPStreamer(
            rtmp_url=stream["rtmp_url"],
            fps=stream.get("fps", 60),
            bitrate=stream.get("bitrate", "8000k"),
            encoder=stream.get("encoder", "h264_nvenc"),
            preset=stream.get("preset", "p4"),
            audio_device=audio.get("device") if audio.get("enabled", False) else None,
            audio_bitrate=audio.get("bitrate", "160k"),
        )
        self.streamer.start()

    def run(self):
        stream_fps = max(1.0, float(self.config.get("stream", {}).get("fps", 60)))
        frame_period = 1.0 / stream_fps
        tracker = self.config["tracker"]
        debug = self.config.get("debug", {})
        capture = CaptureWorker(self.open_camera, self.latest_frame, self.stop_event)
        guidance = GuidanceWorker(self.config, self.latest_frame, self.guidance, self.stop_event)
        capture.start()
        guidance.start()
        self.start_streamer()

        current_crop = None
        current_shape = None
        last_render = time.monotonic()
        next_render = last_render
        measured_frames = 0
        fps_started = last_render
        logger.info("Starting independent 4K capture, guidance, and %.1f fps render paths", stream_fps)

        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                if now < next_render:
                    self.stop_event.wait(next_render - now)
                    continue
                if now - next_render > frame_period * 2:
                    next_render = now
                next_render += frame_period

                snapshot = self.latest_frame.get()
                if snapshot is None:
                    if capture.error:
                        raise capture.error
                    self.stop_event.wait(0.01)
                    continue

                frame = snapshot.frame
                frame_h, frame_w = frame.shape[:2]
                shape = (frame_w, frame_h)
                if current_crop is None or current_shape != shape:
                    current_crop = overview_crop(frame_w, frame_h)
                    current_shape = shape

                target_crop = self.guidance.target_for(frame_w, frame_h, now, tracker.get("hold_seconds", 1.0))
                elapsed = max(0.0, now - last_render)
                last_render = now
                current_crop = advance_crop(
                    current_crop,
                    target_crop,
                    elapsed,
                    frame_w,
                    frame_h,
                    tracker.get("max_zoom_fraction_per_second", 0.35),
                    tracker.get("max_pan_fraction_per_second", 0.25),
                )
                output = crop_and_scale(frame, current_crop)

                if self.streamer:
                    self.streamer.send_frame(output)
                if debug.get("window", False):
                    cv2.imshow("BirdCam", output)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.stop_event.set()

                measured_frames += 1
                if debug.get("log_fps", True) and measured_frames >= round(stream_fps):
                    logged_at = time.monotonic()
                    fps = measured_frames / max(logged_at - fps_started, 0.001)
                    bird_count, guidance_at = self.guidance.status()
                    guidance_age = None if guidance_at is None else logged_at - guidance_at
                    logger.info(
                        "Output FPS: %.1f | Birds: %d | Guidance age: %s | Crop: %s",
                        fps,
                        bird_count,
                        "n/a" if guidance_age is None else f"{guidance_age:.2f}s",
                        current_crop,
                    )
                    measured_frames = 0
                    fps_started = logged_at

                if capture.error:
                    raise capture.error
                if guidance.error:
                    raise guidance.error
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.stop_event.set()
            capture.join(timeout=3)
            guidance.join(timeout=3)
            if self.streamer:
                self.streamer.stop()
            cv2.destroyAllWindows()
            logger.info("BirdCam stopped")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    BirdCam(config).run()


if __name__ == "__main__":
    main()
