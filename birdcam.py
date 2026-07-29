"""BirdCam — Adaptive Zoom Bird Feeder Livestream Tracker."""

import cv2
import numpy as np
import yaml
import time
import sys
import logging

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("birdcam")

OUT_W = 1080
OUT_H = 1920
FULL_W = 3840
FULL_H = 2160


def detect_birds(model, frame, conf_thresh=0.45):
    """Detect birds using YOLOv8. Returns list of (x1, y1, x2, y2)."""
    results = model.predict(frame, verbose=False, conf=conf_thresh)
    birds = []
    for res in results:
        if res.boxes is None:
            continue
        for box, cls in zip(res.boxes.xywh, res.boxes.cls):
            if int(cls) != 16:  # COCO class 16 = bird
                continue
            cx, cy, w, h = box.tolist()
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            birds.append((x1, y1, x2, y2))
    return birds


def lerp(a, b, t):
    return a + (b - a) * t


def clamp_rect(x, y, w, h, frame_w, frame_h):
    """Clamp rectangle to fit within frame."""
    if w < 1:
        w = 1
    if h < 1:
        h = 1
    x = max(0, min(x, frame_w - w))
    y = max(0, min(y, frame_h - h))
    return (x, y, w, h)


def compute_bird_crop(birds, frame_w, frame_h, padding=200):
    """Compute crop region containing all birds with 9:16 aspect ratio."""
    if not birds:
        return None

    all_x1 = min(b[0] for b in birds)
    all_y1 = min(b[1] for b in birds)
    all_x2 = max(b[2] for b in birds)
    all_y2 = max(b[3] for b in birds)
    center_x = (all_x1 + all_x2) / 2
    center_y = (all_y1 + all_y2) / 2

    # Crop size to fit all birds with padding
    needed_w = (all_x2 - all_x1) + 2 * padding
    needed_h = (all_y2 - all_y1) + 2 * padding

    # Enforce 9:16 aspect ratio
    aspect = 1080 / 1920
    if needed_w / needed_h > aspect:
        needed_h = int(needed_w / aspect)
    else:
        needed_w = int(needed_h * aspect)

    # Clamp to frame
    needed_w = min(needed_w, frame_w)
    needed_h = min(needed_h, frame_h)

    x = int(center_x - needed_w / 2)
    y = int(center_y - needed_h / 2)
    return clamp_rect(x, y, needed_w, needed_h, frame_w, frame_h)


def full_frame_view(frame):
    """Scale full 4K frame to 9:16 with black bars."""
    scaled = cv2.resize(frame, (OUT_W, 540), interpolation=cv2.INTER_LINEAR)
    padding_top = (OUT_H - 540) // 2
    result = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
    result[padding_top:padding_top + 540, :, :] = scaled
    return result


def crop_and_scale(frame, crop):
    """Crop a region and scale to 1080x1920."""
    x, y, w, h = crop
    region = frame[y:y + h, x:x + w]
    return cv2.resize(region, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)


class BirdCam:
    """Main application: detect birds, adaptive zoom, stream."""

    def __init__(self, config):
        self.config = config
        self.current_crop = None
        self.hold_timer = 0
        self.streamer = None

    def run(self):
        """Main loop: detect → zoom/pan → output frame."""
        model_path = self.config["detector"]["model_path"]
        logger.info("Loading YOLOv8 model: %s", model_path)
        model = YOLO(model_path)

        device = self.config["camera"]["device"]
        logger.info("Opening camera: %s", device)
        cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FULL_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FULL_H)
        cap.set(cv2.CAP_PROP_FPS, self.config["camera"]["fps"])

        if not cap.isOpened():
            logger.warning("Camera not ready, retrying...")
            time.sleep(1)
            cap.open(device, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FULL_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FULL_H)
            cap.set(cv2.CAP_PROP_FPS, self.config["camera"]["fps"])

        logger.info("Camera opened: %s @ %dx%d",
                    device,
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # Optional: start RTMP streamer
        if self.config.get("stream") and self.config["stream"].get("rtmp_url"):
            from streamer import RTMPStreamer
            self.streamer = RTMPStreamer(
                rtmp_url=self.config["stream"]["rtmp_url"],
                fps=self.config["stream"].get("fps", 30),
                bitrate=self.config["stream"].get("bitrate", "4000k")
            )
            self.streamer.start()

        logger.info("Starting BirdCam...")
        frame_count = 0
        last_fps_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Detect birds
                birds = detect_birds(model, frame, self.config["detector"]["conf_thresh"])

                # Compute output
                if birds:
                    self.hold_timer = 0
                    target_crop = compute_bird_crop(
                        birds, FULL_W, FULL_H, self.config["tracker"]["padding"]
                    )
                    if target_crop is not None:
                        if self.current_crop is not None:
                            # Smooth transition
                            self.current_crop = (
                                int(lerp(self.current_crop[0], target_crop[0], 0.2)),
                                int(lerp(self.current_crop[1], target_crop[1], 0.2)),
                                int(lerp(self.current_crop[2], target_crop[2], 0.2)),
                                int(lerp(self.current_crop[3], target_crop[3], 0.2)),
                            )
                        else:
                            self.current_crop = target_crop

                        # Crop and scale
                        try:
                            output = crop_and_scale(frame, self.current_crop)
                        except Exception:
                            output = full_frame_view(frame)
                    else:
                        output = full_frame_view(frame)
                else:
                    # No birds: hold briefly, then zoom out
                    self.hold_timer += 1
                    if self.hold_timer > self.config["tracker"]["hold_frames"]:
                        self.current_crop = None
                        output = full_frame_view(frame)
                    else:
                        # Holding: keep last crop if available
                        if self.current_crop is not None:
                            try:
                                output = crop_and_scale(frame, self.current_crop)
                            except Exception:
                                output = full_frame_view(frame)
                        else:
                            output = full_frame_view(frame)

                # Stream output
                if self.streamer:
                    self.streamer.send_frame(output)

                # Debug window
                if self.config["debug"]["window"]:
                    cv2.imshow("BirdCam", output)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1
                if frame_count % 60 == 0:
                    elapsed = time.time() - last_fps_time
                    fps = 60 / max(elapsed, 0.1)
                    state = "BIRDS" if birds else "FULL"
                    logger.info("FPS: %.1f | Birds: %d | Mode: %s", fps, len(birds), state)
                    last_fps_time = time.time()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            if self.streamer:
                self.streamer.stop()
            cap.release()
            cv2.destroyAllWindows()
            logger.info("BirdCam stopped")


def main():
    """Entry point."""
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    app = BirdCam(config)
    app.run()


if __name__ == "__main__":
    main()