"""Probe camera formats and measure actual delivered frame rate.

OpenCV cannot reliably enumerate every UVC mode on Windows, so this utility
tries a practical matrix of requested modes and reports what the driver
actually negotiates and delivers.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import cv2


@dataclass(frozen=True)
class ProbeMode:
    fourcc: str
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class ProbeResult:
    backend: str
    requested: ProbeMode
    opened: bool
    negotiated_fourcc: str = ""
    negotiated_width: int = 0
    negotiated_height: int = 0
    reported_fps: float = 0.0
    delivered_fps: float = 0.0
    frames: int = 0
    failures: int = 0
    median_interval_ms: float = 0.0
    p95_interval_ms: float = 0.0
    note: str = ""


def fourcc_to_text(value: float) -> str:
    value = int(value)
    return "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00 ")


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def backend_options(name: str) -> list[tuple[str, int]]:
    options = {
        "dshow": [("DSHOW", cv2.CAP_DSHOW)],
        "msmf": [("MSMF", cv2.CAP_MSMF)],
        "any": [("ANY", cv2.CAP_ANY)],
    }
    if name != "all":
        return options[name]
    if sys.platform == "win32":
        return options["dshow"] + options["msmf"]
    return options["any"]


def default_modes(quick: bool = False) -> list[ProbeMode]:
    if quick:
        return [
            ProbeMode("MJPG", 3840, 2160, 60),
            ProbeMode("YUY2", 3840, 2160, 60),
            ProbeMode("MJPG", 1920, 1080, 60),
        ]
    return [
        ProbeMode(fourcc, width, height, fps)
        for fourcc in ("MJPG", "H264", "YUY2")
        for width, height, fps in (
            (3840, 2160, 60),
            (3840, 2160, 30),
            (2560, 1440, 60),
            (1920, 1080, 60),
            (1920, 1080, 30),
            (1280, 720, 60),
        )
    ]


def configure_capture(cap: cv2.VideoCapture, mode: ProbeMode) -> None:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*mode.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, mode.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, mode.height)
    cap.set(cv2.CAP_PROP_FPS, mode.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def probe_mode(
    device: int,
    backend_name: str,
    backend: int,
    mode: ProbeMode,
    warmup_seconds: float,
    duration_seconds: float,
) -> ProbeResult:
    cap = cv2.VideoCapture(device, backend)
    try:
        if not cap.isOpened():
            return ProbeResult(backend_name, mode, opened=False, note="camera did not open")

        configure_capture(cap, mode)

        negotiated_fourcc = fourcc_to_text(cap.get(cv2.CAP_PROP_FOURCC))
        negotiated_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        negotiated_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported_fps = float(cap.get(cv2.CAP_PROP_FPS))

        warmup_deadline = time.perf_counter() + max(0.0, warmup_seconds)
        while time.perf_counter() < warmup_deadline:
            cap.read()

        started = time.perf_counter()
        deadline = started + max(0.1, duration_seconds)
        timestamps: list[float] = []
        failures = 0
        last_shape: tuple[int, ...] | None = None

        while time.perf_counter() < deadline:
            ok, frame = cap.read()
            timestamp = time.perf_counter()
            if not ok or frame is None:
                failures += 1
                continue
            timestamps.append(timestamp)
            last_shape = frame.shape

        elapsed = max(time.perf_counter() - started, 1e-9)
        frames = len(timestamps)
        intervals = [later - earlier for earlier, later in zip(timestamps, timestamps[1:])]
        note_parts: list[str] = []
        if last_shape is not None:
            actual_height, actual_width = last_shape[:2]
            if (actual_width, actual_height) != (negotiated_width, negotiated_height):
                note_parts.append(f"decoded {actual_width}x{actual_height}")
        if negotiated_fourcc != mode.fourcc:
            note_parts.append(f"format fallback from {mode.fourcc}")
        if (negotiated_width, negotiated_height) != (mode.width, mode.height):
            note_parts.append("resolution fallback")
        if frames == 0:
            note_parts.append("no frames delivered")

        return ProbeResult(
            backend=backend_name,
            requested=mode,
            opened=True,
            negotiated_fourcc=negotiated_fourcc or "?",
            negotiated_width=negotiated_width,
            negotiated_height=negotiated_height,
            reported_fps=reported_fps,
            delivered_fps=frames / elapsed,
            frames=frames,
            failures=failures,
            median_interval_ms=statistics.median(intervals) * 1000 if intervals else 0.0,
            p95_interval_ms=percentile(intervals, 0.95) * 1000 if intervals else 0.0,
            note="; ".join(note_parts),
        )
    finally:
        cap.release()


def format_result(result: ProbeResult) -> str:
    requested = result.requested
    request_text = f"{requested.fourcc} {requested.width}x{requested.height}@{requested.fps:g}"
    if not result.opened:
        return f"{result.backend:5} | {request_text:24} | OPEN FAILED | {result.note}"
    negotiated = (
        f"{result.negotiated_fourcc} "
        f"{result.negotiated_width}x{result.negotiated_height}@{result.reported_fps:g}"
    )
    return (
        f"{result.backend:5} | {request_text:24} | {negotiated:24} | "
        f"actual {result.delivered_fps:6.2f} fps | "
        f"median {result.median_interval_ms:6.2f} ms | "
        f"p95 {result.p95_interval_ms:6.2f} ms | "
        f"fail {result.failures:3d}"
        + (f" | {result.note}" if result.note else "")
    )


def parse_modes(values: Iterable[str]) -> list[ProbeMode]:
    modes: list[ProbeMode] = []
    for value in values:
        try:
            fourcc, dimensions, fps_text = value.split(":")
            width_text, height_text = dimensions.lower().split("x")
            modes.append(ProbeMode(fourcc.upper(), int(width_text), int(height_text), float(fps_text)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid mode {value!r}; expected FOURCC:WIDTHxHEIGHT:FPS"
            ) from exc
        if len(modes[-1].fourcc) != 4:
            raise argparse.ArgumentTypeError("FOURCC must contain exactly four characters")
    return modes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe requested camera modes and measure actual delivered FPS."
    )
    parser.add_argument("--device", type=int, default=0, help="OpenCV camera device index")
    parser.add_argument(
        "--backend",
        choices=("all", "dshow", "msmf", "any"),
        default="all" if sys.platform == "win32" else "any",
        help="OpenCV capture backend to test",
    )
    parser.add_argument("--warmup", type=float, default=1.0, help="Warm-up seconds per mode")
    parser.add_argument("--duration", type=float, default=3.0, help="Measurement seconds per mode")
    parser.add_argument("--quick", action="store_true", help="Probe only three likely modes")
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        metavar="FOURCC:WIDTHxHEIGHT:FPS",
        help="Probe a custom mode; may be repeated",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        modes = parse_modes(args.mode) if args.mode else default_modes(args.quick)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc

    print("BirdCam camera diagnostic")
    print("Reported FPS is driver metadata; actual FPS is measured from completed reads.")
    print("OpenCV cannot enumerate every UVC mode, so unsupported requests may fall back silently.\n")

    results: list[ProbeResult] = []
    for backend_name, backend in backend_options(args.backend):
        for mode in modes:
            print(f"Probing {backend_name} {mode.fourcc} {mode.width}x{mode.height}@{mode.fps:g} ...")
            result = probe_mode(
                args.device,
                backend_name,
                backend,
                mode,
                args.warmup,
                args.duration,
            )
            results.append(result)
            print(format_result(result))

    usable = [result for result in results if result.opened and result.frames > 0]
    if usable:
        best = max(usable, key=lambda result: result.delivered_fps)
        print("\nFastest measured mode:")
        print(format_result(best))
    else:
        print("\nNo mode delivered frames.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
