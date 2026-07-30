"""Photo-based camera emulator for BirdCam development and tests."""

from __future__ import annotations

import glob
import time
from pathlib import Path

import cv2


class PhotoSequenceCapture:
    """Expose a folder of still images through the VideoCapture-style API.

    Each photo is held for ``seconds_per_photo`` while ``read`` emits frames at
    ``fps``. The sequence loops by default and can be paced in real time.
    Optional output dimensions center-crop and resize each photo so performance
    tests exercise the same source-frame size as the configured camera.
    """

    def __init__(
        self,
        pattern: str,
        fps: float = 60.0,
        seconds_per_photo: float = 2.0,
        loop: bool = True,
        realtime: bool = True,
        width: int | None = None,
        height: int | None = None,
        clock=time.monotonic,
        sleep=time.sleep,
    ):
        self.paths = [Path(path) for path in sorted(glob.glob(pattern))]
        if not self.paths:
            raise ValueError(f"No photos matched {pattern!r}")
        if (width is None) != (height is None):
            raise ValueError("width and height must be supplied together")
        if width is not None and (int(width) < 1 or int(height) < 1):
            raise ValueError("width and height must be positive")

        self.fps = max(0.1, float(fps))
        self.frame_period = 1.0 / self.fps
        self.frames_per_photo = max(1, round(self.fps * max(0.0, seconds_per_photo)))
        self.loop = loop
        self.realtime = realtime
        self.output_size = None if width is None else (int(width), int(height))
        self.clock = clock
        self.sleep = sleep
        self.index = 0
        self.frame_number = 0
        self.next_frame_at = None
        self.frame = None
        self.opened = True
        self._load_current()

    def _prepare_frame(self, frame):
        if self.output_size is None:
            return frame

        target_w, target_h = self.output_size
        source_h, source_w = frame.shape[:2]
        source_aspect = source_w / source_h
        target_aspect = target_w / target_h

        if source_aspect > target_aspect:
            crop_w = max(1, round(source_h * target_aspect))
            x = (source_w - crop_w) // 2
            frame = frame[:, x:x + crop_w]
        elif source_aspect < target_aspect:
            crop_h = max(1, round(source_w / target_aspect))
            y = (source_h - crop_h) // 2
            frame = frame[y:y + crop_h, :]

        interpolation = (
            cv2.INTER_AREA
            if frame.shape[1] >= target_w and frame.shape[0] >= target_h
            else cv2.INTER_LINEAR
        )
        return cv2.resize(frame, (target_w, target_h), interpolation=interpolation)

    def _load_current(self):
        frame = cv2.imread(str(self.paths[self.index]), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Unable to decode photo {self.paths[self.index]}")
        self.frame = self._prepare_frame(frame)

    def isOpened(self):
        return self.opened

    def _pace(self):
        now = self.clock()
        if self.next_frame_at is None:
            self.next_frame_at = now
        delay = self.next_frame_at - now
        if delay > 0:
            self.sleep(delay)
            now = self.clock()
        # A live camera does not replay missed deadlines. Reset after a stall so
        # the capture worker cannot publish a burst of synthetic catch-up frames.
        if now - self.next_frame_at > self.frame_period:
            self.next_frame_at = now
        self.next_frame_at += self.frame_period

    def read(self):
        if not self.opened:
            return False, None
        if self.realtime:
            self._pace()

        output = self.frame.copy()
        self.frame_number += 1
        if self.frame_number >= self.frames_per_photo:
            self.frame_number = 0
            next_index = self.index + 1
            if next_index >= len(self.paths):
                if not self.loop:
                    self.release()
                    return True, output
                next_index = 0
            self.index = next_index
            self._load_current()
        return True, output

    def release(self):
        self.opened = False

    def get(self, prop):
        if self.frame is None:
            return 0.0
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.frame.shape[1])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.frame.shape[0])
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return 0.0

    def set(self, _prop, _value):
        return False
