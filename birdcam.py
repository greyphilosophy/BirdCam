"""BirdCam — Adaptive Zoom Bird Feeder Livestream Tracker."""

import logging
import sys
import time
from typing import Iterable

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


def detect_birds(model, frame, conf_thresh=0.45, imgsz=1280, device=0):
    """Detect birds and return bounding boxes as ``(x1, y1, x2, y2)``."""
    results = model.predict(
        frame,
        verbose=False,
        conf=conf_thresh,
        imgsz=imgsz,
        device=device,
    )
    birds = []
    for result in results:
        if result.boxes is None:
            continue
        for box, class_id in zip(result.boxes.xyxy, result.boxes.cls):
            if int(class_id) == BIRD_CLASS_ID:
                birds.append(tuple(int(value) for value in box.tolist()))
    return birds


def lerp(a, b, amount):
    return a + (b - a) * amount


def clamp_rect(x, y, width, height, frame_w, frame_h):
    """Clamp a rectangle to the frame without changing its dimensions."""
    width = max(1, min(int(width), frame_w))
    height = max(1, min(int(height), frame_h))
    x = max(0, min(int(x), frame_w - width))
    y = max(0, min(int(y), frame_h - height))
    return x, y, width, height


def compute_bird_crop(birds: Iterable[tuple[int, int, int, int]], frame_w, frame_h, padding=200):
    """Return the smallest valid 9:16 crop containing all birds, or ``None``."""
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

    max_h = min(frame_h, frame_w / OUT_ASPECT)
    max_w = max_h * OUT_ASPECT
    if crop_w > max_w or crop_h > max_h:
        return None

    crop_w = max(1, int(round(crop_w)))
    crop_h = max(1, int(round(crop_w / OUT_ASPECT)))
    if crop_h > frame_h:
        crop_h = frame_h
        crop_w = int(round(crop_h * OUT_ASPECT))

    x = int(round(center_x - crop_w / 2))
    y = int(round(center_y - crop_h / 2))
    return clamp_rect(x, y, crop_w, crop_h, frame_w, frame_h)


def full_frame_view(frame):
    """Letterbox the full source frame into the vertical output frame."""
    frame_h, frame_w = frame.shape[:2]
    scale = min(OUT_W / frame_w, OUT_H / frame_h)
    scaled_w = max(1, int(round(frame_w * scale)))
    scaled_h = max(1, int(round(frame_h * scale)))
    scaled = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    result = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
    x = (OUT_W - scaled_w) // 2
    y = (OUT_H - scaled_h) // 2
    result[y:y + scaled_h, x:x + scaled_w] = scaled
    return result


def crop_and_scale(frame, crop):
    """Crop a valid 9:16 region and scale it to 1080x1920."""
    x, y, width, height = crop
    region = frame[y:y + height, x:x + width]
    if region.size == 0:
        raise ValueError(f"Empty crop: {crop}")
    return cv2.resize(region, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)


def fourcc_to_text(value):
    value = int(value)
    return "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4))


class BirdCam:
    """Detect birds, generate an adaptive crop, and stream the result."""

    def __init__(self, config):
        self.config = config
        self.current_crop = None
        self.last_bird_time = None
        self.streamer = None
        self.capture_failures = 0

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
        logger.info(
            "Camera opened: %s @ %dx%d %.2f fps (%s)",
            device,
            actual_w,
            actual_h,
            actual_fps,
            actual_fourcc,
        )

        requested = (
            camera.get("width", DEFAULT_FULL_W),
            camera.get("height", DEFAULT_FULL_H),
            camera.get("fps", 60),
        )
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

        self.streamer = RTMPStreamer(
            rtmp_url=stream["rtmp_url"],
            fps=stream.get("fps", 60),
            bitrate=stream.get("bitrate", "8000k"),
            encoder=stream.get("encoder", "h264_nvenc"),
            preset=stream.get("preset", "p4"),
        )
        self.streamer.start()

    def render(self, frame, birds, now):
        frame_h, frame_w = frame.shape[:2]
        tracker = self.config["tracker"]

        if birds:
            self.last_bird_time = now
            target_crop = compute_bird_crop(birds, frame_w, frame_h, tracker["padding"])
            if target_crop is None:
                self.current_crop = None
                return full_frame_view(frame)

            smoothing = tracker.get("smoothing", 0.2)
            if self.current_crop is None:
                self.current_crop = target_crop
            else:
                self.current_crop = tuple(
                    int(round(lerp(current, target, smoothing)))
                    for current, target in zip(self.current_crop, target_crop)
                )
                self.current_crop = clamp_rect(*self.current_crop, frame_w, frame_h)
        elif self.last_bird_time is None or now - self.last_bird_time > tracker.get("hold_seconds", 1.0):
            self.current_crop = None

        if self.current_crop is None:
            return full_frame_view(frame)

        try:
            return crop_and_scale(frame, self.current_crop)
        except (ValueError, cv2.error) as exc:
            logger.warning("Invalid crop %s: %s", self.current_crop, exc)
            self.current_crop = None
            return full_frame_view(frame)

    def run(self):
        detector = self.config["detector"]
        logger.info("Loading YOLO model: %s", detector["model_path"])
        model = YOLO(detector["model_path"])
        cap = self.open_camera()
        self.start_streamer()

        frame_count = 0
        measured_frames = 0
        last_fps_time = time.monotonic()
        birds = []
        detection_interval = max(1, detector.get("interval_frames", 1))
        logger.info("Starting BirdCam")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    self.capture_failures += 1
                    if self.capture_failures < 10:
                        time.sleep(0.01)
                        continue
                    logger.warning("Camera read failed repeatedly; reconnecting")
                    cap.release()
                    time.sleep(1)
                    cap = self.open_camera()
                    self.capture_failures = 0
                    continue
                self.capture_failures = 0

                if frame_count % detection_interval == 0:
                    birds = detect_birds(
                        model,
                        frame,
                        detector["conf_thresh"],
                        detector.get("imgsz", 1280),
                        detector.get("device", 0),
                    )

                output = self.render(frame, birds, time.monotonic())
                if self.streamer:
                    self.streamer.send_frame(output)

                debug = self.config.get("debug", {})
                if debug.get("window", False):
                    cv2.imshow("BirdCam", output)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1
                measured_frames += 1
                if debug.get("log_fps", True) and measured_frames >= 60:
                    now = time.monotonic()
                    fps = measured_frames / max(now - last_fps_time, 0.001)
                    logger.info("FPS: %.1f | Birds: %d | Mode: %s", fps, len(birds), "BIRDS" if birds else "FULL")
                    measured_frames = 0
                    last_fps_time = now
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            if self.streamer:
                self.streamer.stop()
            cap.release()
            cv2.destroyAllWindows()
            logger.info("BirdCam stopped")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    BirdCam(config).run()


if __name__ == "__main__":
    main()
