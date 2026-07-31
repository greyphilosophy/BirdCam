"""FFmpeg RTMP streaming for BirdCam."""

import logging
import subprocess
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger("birdcam.streamer")

OUT_W = 1080
OUT_H = 1920


class RTMPStreamer:
    """Stream raw BGR frames and optional microphone audio through FFmpeg."""

    def __init__(
        self,
        rtmp_url: str,
        fps: int = 60,
        bitrate: str = "8000k",
        encoder: str = "h264_nvenc",
        preset: str = "p4",
        audio_device: str | None = None,
        audio_bitrate: str = "160k",
    ):
        self.rtmp_url = rtmp_url
        self.fps = fps
        self.bitrate = bitrate
        self.encoder = encoder
        self.preset = preset
        self.audio_device = audio_device
        self.audio_bitrate = audio_bitrate
        self.process = None
        self._metrics_lock = threading.Lock()
        self._metrics_started = time.monotonic()
        self._sent_frames = 0
        self._write_seconds = 0.0
        self._maximum_write_seconds = 0.0

    def _command(self):
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-fflags",
            "nobuffer",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{OUT_W}x{OUT_H}",
            "-framerate",
            str(self.fps),
            "-i",
            "pipe:0",
        ]

        if self.audio_device:
            command.extend([
                "-f",
                "dshow",
                "-i",
                f"audio={self.audio_device}",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
            ])
        else:
            command.append("-an")

        command.extend([
            "-c:v",
            self.encoder,
            "-flags:v",
            "+low_delay",
        ])
        if self.encoder == "libx264":
            command.extend([
                "-threads",
                "4",
                "-preset",
                self.preset or "veryfast",
                "-tune",
                "zerolatency",
            ])
        else:
            command.extend([
                "-preset",
                self.preset,
                "-tune",
                "ll",
            ])

        command.extend([
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            self.bitrate,
            "-maxrate",
            self.bitrate,
            "-bufsize",
            self.bitrate,
            "-g",
            str(self.fps * 2),
            "-flush_packets",
            "1",
        ])

        if self.audio_device:
            command.extend([
                "-c:a",
                "aac",
                "-b:a",
                self.audio_bitrate,
                "-ar",
                "48000",
            ])

        command.extend(["-f", "flv", self.rtmp_url])
        return command

    def _safe_command_text(self, command):
        """Render the FFmpeg command without exposing the RTMP stream key."""
        safe_command = list(command)
        if safe_command and safe_command[-1] == self.rtmp_url:
            safe_command[-1] = "<redacted-rtmp-url>"
        return subprocess.list2cmdline(safe_command)

    def _record_write(self, elapsed_seconds):
        with self._metrics_lock:
            self._sent_frames += 1
            self._write_seconds += elapsed_seconds
            self._maximum_write_seconds = max(self._maximum_write_seconds, elapsed_seconds)

    def metrics(self, now=None, reset=True):
        """Return FFmpeg input throughput and blocking-write latency metrics."""
        now = time.monotonic() if now is None else now
        with self._metrics_lock:
            elapsed = max(now - self._metrics_started, 0.001)
            frames = self._sent_frames
            result = {
                "fps": frames / elapsed,
                "frames": frames,
                "average_write_ms": 1000.0 * self._write_seconds / max(frames, 1),
                "maximum_write_ms": 1000.0 * self._maximum_write_seconds,
            }
            if reset:
                self._metrics_started = now
                self._sent_frames = 0
                self._write_seconds = 0.0
                self._maximum_write_seconds = 0.0
        return result

    def start(self):
        """Start FFmpeg and leave stderr attached so failures stay visible."""
        if self.process and self.process.poll() is None:
            return
        command = self._command()
        logger.info("Starting FFmpeg: %s", self._safe_command_text(command))
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=None,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("FFmpeg was not found on PATH") from exc
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg stdin was not created")
        with self._metrics_lock:
            self._metrics_started = time.monotonic()
            self._sent_frames = 0
            self._write_seconds = 0.0
            self._maximum_write_seconds = 0.0

    def send_frame(self, frame: np.ndarray):
        """Send one BGR frame, restarting FFmpeg after a broken process."""
        if self.process is None or self.process.poll() is not None:
            exit_code = None if self.process is None else self.process.returncode
            logger.warning("FFmpeg is not running (exit code %s); restarting", exit_code)
            self.start()

        if frame.shape[:2] != (OUT_H, OUT_W):
            frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        payload = frame.tobytes()

        write_started = time.perf_counter()
        try:
            self.process.stdin.write(payload)
        except (BrokenPipeError, OSError) as exc:
            logger.error("FFmpeg pipe failed: %s", exc)
            self.stop()
            self.start()
            write_started = time.perf_counter()
            self.process.stdin.write(payload)
        self._record_write(time.perf_counter() - write_started)

        with self._metrics_lock:
            should_log = self._sent_frames >= max(1, round(self.fps))
        if should_log:
            metrics = self.metrics()
            logger.info(
                "FFmpeg input FPS: %.1f | Pipe write: %.2f ms avg / %.2f ms max",
                metrics["fps"],
                metrics["average_write_ms"],
                metrics["maximum_write_ms"],
            )

    def stop(self):
        """Close FFmpeg without waiting forever for a stalled process."""
        process = self.process
        self.process = None
        if not process:
            return
        if process.stdin:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg did not exit cleanly; terminating")
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        logger.info("FFmpeg pipeline stopped")
