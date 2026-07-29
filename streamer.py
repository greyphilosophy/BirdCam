"""FFmpeg RTMP streaming for BirdCam.

Pipes cropped frames into an FFmpeg subprocess that outputs to an RTMP URL
(TikTok livestream).
"""

import logging
import subprocess

import cv2
import numpy as np

logger = logging.getLogger("birdcam.streamer")

# 9:16 output dimensions
OUT_W = 1080
OUT_H = 1920


class RTMPStreamer:
    """Stream frames to an RTMP URL via FFmpeg pipe."""

    def __init__(self, rtmp_url: str, fps: int = 30, bitrate: str = "4000k"):
        self.rtmp_url = rtmp_url
        self.fps = fps
        self.bitrate = bitrate
        self.process = None

    def start(self):
        """Start the FFmpeg subprocess, feeding raw YUYV via stdin."""
        cmd = [
            "ffmpeg",
            "-y",
            "-thread_num", "4",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{OUT_W}x{OUT_H}",
            "-framerate", str(self.fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "medium",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-b:v", self.bitrate,
            "-maxrate", self.bitrate,
            "-g", str(self.fps * 2),
            "-f", "flv",
            self.rtmp_url,
        ]
        logger.info("Starting FFmpeg: %s", " ".join(cmd))
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=2**20,
        )
        logger.info("FFmpeg pipeline started")

    def send_frame(self, frame: np.ndarray):
        """Send a BGR frame to the FFmpeg pipe."""
        if frame.shape[:2] != (OUT_H, OUT_W):
            frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
        self.process.stdin.write(frame.tobytes())

    def stop(self):
        """Stop the FFmpeg pipeline."""
        if self.process:
            self.process.stdin.close()
            self.process.wait()
            logger.info("FFmpeg pipeline stopped")
