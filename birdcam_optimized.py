"""Optimized BirdCam pipeline that avoids rotating every native 4K frame."""

import sys
import time

import cv2
import numpy as np
import yaml

import birdcam_legacy as legacy
from birdcam_legacy import *  # noqa: F401,F403

# Ultralytics COCO ordering: bird=14, cat=15, dog=16.
BIRD_CLASS_ID = 14
legacy.BIRD_CLASS_ID = BIRD_CLASS_ID
logger = legacy.logger


def rotated_frame_size(frame_w, frame_h, degrees):
    """Return logical dimensions after a supported clockwise rotation."""
    normalized = int(degrees) % 360
    if normalized not in {0, 90, 180, 270}:
        raise ValueError("camera.rotation_degrees must be 0, 90, 180, or 270")
    return (frame_h, frame_w) if normalized in {90, 270} else (frame_w, frame_h)


def native_crop_for_rotated_crop(crop, frame_w, frame_h, degrees):
    """Map a crop in rotated coordinates back onto the native camera frame."""
    x, y, width, height = crop
    normalized = int(degrees) % 360
    if normalized == 0:
        native = (x, y, width, height)
    elif normalized == 90:
        native = (y, frame_h - x - width, height, width)
    elif normalized == 180:
        native = (
            frame_w - x - width,
            frame_h - y - height,
            width,
            height,
        )
    elif normalized == 270:
        native = (frame_w - y - height, x, height, width)
    else:
        raise ValueError("camera.rotation_degrees must be 0, 90, 180, or 270")
    return legacy.clamp_rect(*native, frame_w, frame_h)


def crop_and_scale_rotated(frame, crop, degrees):
    """Render an upright crop while rotating only output-sized pixels."""
    native_h, native_w = frame.shape[:2]
    source_crop = native_crop_for_rotated_crop(
        crop, native_w, native_h, degrees
    )
    x, y, width, height = source_crop
    region = frame[y:y + height, x:x + width]
    if region.size == 0:
        raise ValueError(f"Empty native crop: {source_crop}")

    rotated_width = crop[2]
    rotated_height = crop[3]
    if legacy._aspect_ratios_match(
        rotated_width, rotated_height, legacy.OUT_W, legacy.OUT_H
    ):
        render_w, render_h = legacy.OUT_W, legacy.OUT_H
    else:
        scale = min(
            legacy.OUT_W / rotated_width,
            legacy.OUT_H / rotated_height,
        )
        render_w = min(
            legacy.OUT_W,
            max(1, int(round(rotated_width * scale))),
        )
        render_h = min(
            legacy.OUT_H,
            max(1, int(round(rotated_height * scale))),
        )

    normalized = int(degrees) % 360
    resize_size = (
        (render_h, render_w)
        if normalized in {90, 270}
        else (render_w, render_h)
    )
    resized = cv2.resize(
        region,
        resize_size,
        interpolation=cv2.INTER_LINEAR,
    )
    rendered = legacy.rotate_frame(resized, normalized)
    if (render_w, render_h) == (legacy.OUT_W, legacy.OUT_H):
        return rendered

    output = np.zeros(
        (legacy.OUT_H, legacy.OUT_W, 3),
        dtype=np.uint8,
    )
    offset_x = (legacy.OUT_W - render_w) // 2
    offset_y = (legacy.OUT_H - render_h) // 2
    output[
        offset_y:offset_y + render_h,
        offset_x:offset_x + render_w,
    ] = rendered
    return output


class CaptureWorker(legacy.CaptureWorker):
    """Capture native frames without a full-resolution rotation copy."""

    def run(self):
        cap = None
        failures = 0
        try:
            while not self.stop_event.is_set():
                if cap is None:
                    cap = self.open_camera()
                    with self._metrics_lock:
                        self._captured_frames = 0
                        self._metrics_started = time.monotonic()
                ok, frame = cap.read()
                if ok:
                    failures = 0
                    self.latest_frame.publish(frame)
                    with self._metrics_lock:
                        self._captured_frames += 1
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


class GuidanceWorker(legacy.GuidanceWorker):
    """Rotate only sampled guidance frames before YOLO inference."""

    def __init__(
        self,
        config,
        latest_frame,
        guidance,
        stop_event,
        rotation_degrees=0,
    ):
        super().__init__(config, latest_frame, guidance, stop_event)
        self.rotation_degrees = rotation_degrees

    def _on_detection(self, birds, frame_w, frame_h, tracker):
        """Allow specialized guidance workers to observe each detector result."""

    def run(self):
        detector = self.config["detector"]
        tracker = self.config["tracker"]
        try:
            logger.info("Loading YOLO guidance model: %s", detector["model_path"])
            model = legacy.YOLO(detector["model_path"])
            sample_period = 1.0 / max(
                0.1,
                float(detector.get("max_fps", 5.0)),
            )
            sequence = 0
            next_sample = 0.0
            while not self.stop_event.is_set():
                snapshot = self.latest_frame.wait_for_newer(
                    sequence,
                    timeout=0.25,
                )
                if snapshot is None:
                    continue
                sequence = snapshot.sequence
                if time.monotonic() < next_sample:
                    continue

                guidance_frame = legacy.rotate_frame(
                    snapshot.frame,
                    self.rotation_degrees,
                )
                birds = legacy.detect_birds(
                    model,
                    guidance_frame,
                    detector["conf_thresh"],
                    detector.get("imgsz", 1280),
                    detector.get("device", 0),
                )
                frame_h, frame_w = guidance_frame.shape[:2]
                self._on_detection(birds, frame_w, frame_h, tracker)
                self.guidance.publish(
                    legacy.compute_bird_crop(
                        birds,
                        frame_w,
                        frame_h,
                        tracker["padding"],
                    ),
                    len(birds),
                    snapshot.captured_at,
                    published_at=time.monotonic(),
                )
                next_sample = time.monotonic() + sample_period
        except Exception as exc:
            self.error = exc
            logger.exception("Guidance worker stopped")
            self.stop_event.set()


class BirdCam(legacy.BirdCam):
    """Run native capture with sampled rotation and output-sized rendering."""

    def run(self):
        stream_fps = max(
            1.0,
            float(self.config.get("stream", {}).get("fps", 60)),
        )
        frame_period = 1.0 / stream_fps
        camera = self.config["camera"]
        rotation_degrees = camera.get("rotation_degrees", 90)
        tracker = self.config["tracker"]
        idle_view = self.config.get("idle_view", {})
        idle_enabled = idle_view.get("enabled", True)
        idle_after_seconds = idle_view.get("delay_seconds", 3.0)
        debug = self.config.get("debug", {})
        preview_enabled = bool(debug.get("window", False))
        preview_rotation = debug.get("preview_rotation", "none")
        if preview_enabled:
            preview_width, preview_height = legacy.configure_debug_window(debug)
            logger.info(
                "Debug preview: %dx%d window, rotation=%s",
                preview_width,
                preview_height,
                preview_rotation,
            )

        capture = CaptureWorker(
            self.open_camera,
            self.latest_frame,
            self.stop_event,
            rotation_degrees=rotation_degrees,
        )
        guidance = GuidanceWorker(
            self.config,
            self.latest_frame,
            self.guidance,
            self.stop_event,
            rotation_degrees=rotation_degrees,
        )
        capture.start()
        guidance.start()
        self.start_streamer()

        current_crop = None
        current_shape = None
        view_mode = "idle" if idle_enabled else "overview"
        last_render = None
        next_render = time.monotonic()
        last_sequence = 0
        output = None
        measured_frames = resized_frames = 0
        resize_seconds = 0.0
        fps_started = None
        logger.info(
            "Starting native capture, sampled guidance rotation, and %.1f fps rendering; "
            "source rotation=%s°",
            stream_fps,
            int(rotation_degrees) % 360,
        )

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
                if fps_started is None:
                    fps_started = now
                    next_render = now + frame_period

                frame = snapshot.frame
                native_h, native_w = frame.shape[:2]
                frame_w, frame_h = rotated_frame_size(
                    native_w,
                    native_h,
                    rotation_degrees,
                )
                shape = (frame_w, frame_h)
                if current_crop is None or current_shape != shape:
                    current_crop = (
                        legacy.full_frame_crop(frame_w, frame_h)
                        if idle_enabled
                        else legacy.overview_crop(frame_w, frame_h)
                    )
                    current_shape = shape
                    last_render = now
                    output = None

                target_crop, view_mode = self.guidance.view_for(
                    frame_w,
                    frame_h,
                    now,
                    tracker.get("hold_seconds", 1.0),
                    idle_enabled=idle_enabled,
                    idle_after_seconds=idle_after_seconds,
                )
                elapsed = (
                    0.0
                    if last_render is None
                    else max(0.0, now - last_render)
                )
                last_render = now
                next_crop = legacy.advance_crop(
                    current_crop,
                    target_crop,
                    elapsed,
                    frame_w,
                    frame_h,
                    tracker.get("max_zoom_fraction_per_second", 0.35),
                    tracker.get("max_pan_fraction_per_second", 0.25),
                )
                needs_resize = (
                    output is None
                    or snapshot.sequence != last_sequence
                    or next_crop != current_crop
                )
                current_crop = next_crop
                if needs_resize:
                    resize_started = time.perf_counter()
                    output = crop_and_scale_rotated(
                        frame,
                        current_crop,
                        rotation_degrees,
                    )
                    resize_seconds += time.perf_counter() - resize_started
                    resized_frames += 1
                    last_sequence = snapshot.sequence

                if self.streamer:
                    self.streamer.send_frame(output)
                if preview_enabled:
                    cv2.imshow(
                        legacy.DEBUG_WINDOW_NAME,
                        legacy.prepare_debug_preview(
                            output,
                            preview_rotation,
                        ),
                    )
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.stop_event.set()

                measured_frames += 1
                if (
                    debug.get("log_fps", True)
                    and measured_frames >= round(stream_fps)
                ):
                    logged_at = time.monotonic()
                    fps = measured_frames / max(
                        logged_at - fps_started,
                        0.001,
                    )
                    capture_fps = capture.metrics(logged_at)
                    resize_ms = (
                        1000.0
                        * resize_seconds
                        / max(resized_frames, 1)
                    )
                    bird_count, guidance_at = self.guidance.status()
                    guidance_age = (
                        None
                        if guidance_at is None
                        else logged_at - guidance_at
                    )
                    logger.info(
                        "Output FPS: %.1f | Capture FPS: %.1f | Resize: %.2f ms (%d/%d frames) | "
                        "View: %s | Birds: %d | Guidance age: %s | Crop: %s",
                        fps,
                        capture_fps,
                        resize_ms,
                        resized_frames,
                        measured_frames,
                        view_mode,
                        bird_count,
                        "n/a"
                        if guidance_age is None
                        else f"{guidance_age:.2f}s",
                        current_crop,
                    )
                    measured_frames = resized_frames = 0
                    resize_seconds = 0.0
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
