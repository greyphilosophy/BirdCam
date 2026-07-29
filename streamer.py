"""FFmpeg RTMP streaming for BirdCam."""

import logging
import subprocess

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

    def _command(self):
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
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

        command.extend(["-c:v", self.encoder])
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

    def send_frame(self, frame: np.ndarray):
        """Send one BGR frame, restarting FFmpeg after a broken process."""
        if self.process is None or self.process.poll() is not None:
            exit_code = None if self.process is None else self.process.returncode
            logger.warning("FFmpeg is not running (exit code %s); restarting", exit_code)
            self.start()

        if frame.shape[:2] != (OUT_H, OUT_W):
            frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        try:
            self.process.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as exc:
            logger.error("FFmpeg pipe failed: %s", exc)
            self.stop()
            self.start()
            self.process.stdin.write(frame.tobytes())

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