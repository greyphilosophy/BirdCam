import argparse

import pytest

from scripts.camera_diagnostic import (
    ProbeMode,
    ProbeResult,
    backend_options,
    format_result,
    parse_modes,
    percentile,
)


def test_parse_modes_accepts_custom_probe_mode():
    assert parse_modes(["mjpg:3840x2160:60"]) == [ProbeMode("MJPG", 3840, 2160, 60.0)]


def test_parse_modes_rejects_invalid_fourcc():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_modes(["MJG:3840x2160:60"])


def test_percentile_handles_empty_and_sorted_values():
    assert percentile([], 0.95) == 0.0
    assert percentile([0.01, 0.02, 0.03, 0.04], 0.95) == 0.04


def test_format_result_shows_reported_and_measured_modes():
    result = ProbeResult(
        backend="DSHOW",
        requested=ProbeMode("MJPG", 3840, 2160, 60),
        opened=True,
        negotiated_fourcc="YUY2",
        negotiated_width=3840,
        negotiated_height=2160,
        reported_fps=60,
        delivered_fps=44.8,
        frames=135,
        failures=0,
        median_interval_ms=22.1,
        p95_interval_ms=25.0,
        note="format fallback from MJPG",
    )

    text = format_result(result)

    assert "MJPG 3840x2160@60" in text
    assert "YUY2 3840x2160@60" in text
    assert "actual  44.80 fps" in text
    assert "format fallback" in text


def test_backend_any_is_available_cross_platform():
    assert backend_options("any")[0][0] == "ANY"
