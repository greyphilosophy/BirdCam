from streamer import RTMPStreamer


def _last_input_index(command):
    return max(index for index, value in enumerate(command) if value == "-i")


def test_nvenc_command_uses_low_latency_options():
    command = RTMPStreamer("rtmp://example/live", encoder="h264_nvenc", preset="p4")._command()

    assert command[command.index("-c:v") + 1] == "h264_nvenc"
    assert command[command.index("-preset") + 1] == "p4"
    assert command[command.index("-tune") + 1] == "ll"
    assert command[command.index("-fflags") + 1] == "nobuffer"
    assert command[command.index("-flags:v") + 1] == "+low_delay"
    assert command.index("-flags:v") > _last_input_index(command)
    assert command[command.index("-flush_packets") + 1] == "1"
    assert "-threads" not in command
    assert "-an" in command


def test_libx264_command_has_valid_thread_and_tune_arguments():
    command = RTMPStreamer("rtmp://example/live", encoder="libx264", preset="veryfast")._command()

    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-flags:v") + 1] == "+low_delay"
    assert command.index("-flags:v") > _last_input_index(command)
    assert command[command.index("-threads") + 1] == "4"
    assert command[command.index("-preset") + 1] == "veryfast"
    assert command[command.index("-tune") + 1] == "zerolatency"


def test_external_microphone_adds_directshow_audio_input():
    command = RTMPStreamer(
        "rtmp://example/live/key",
        audio_device="Microphone (USB Audio Device)",
    )._command()

    assert "-an" not in command
    assert command[command.index("-map") + 1] == "0:v:0"
    second_map = command.index("-map", command.index("-map") + 1)
    assert command[second_map + 1] == "1:a:0"
    assert "audio=Microphone (USB Audio Device)" in command
    assert command[command.index("-c:a") + 1] == "aac"
    assert command.index("-flags:v") > _last_input_index(command)


def test_safe_command_redacts_rtmp_url():
    streamer = RTMPStreamer("rtmp://example/live/secret-key")
    safe_text = streamer._safe_command_text(streamer._command())

    assert "secret-key" not in safe_text
    assert "<redacted-rtmp-url>" in safe_text


def test_metrics_report_average_maximum_and_fps():
    streamer = RTMPStreamer("rtmp://example/live", fps=60)
    streamer._metrics_started = 10.0
    streamer._record_write(0.004)
    streamer._record_write(0.010)

    metrics = streamer.metrics(now=10.5, reset=False)

    assert metrics["frames"] == 2
    assert metrics["fps"] == 4.0
    assert metrics["average_write_ms"] == 7.0
    assert metrics["maximum_write_ms"] == 10.0


def test_metrics_reset_after_snapshot():
    streamer = RTMPStreamer("rtmp://example/live")
    streamer._metrics_started = 1.0
    streamer._record_write(0.005)

    streamer.metrics(now=2.0, reset=True)
    metrics = streamer.metrics(now=3.0, reset=False)

    assert metrics["frames"] == 0
    assert metrics["fps"] == 0.0
    assert metrics["average_write_ms"] == 0.0
    assert metrics["maximum_write_ms"] == 0.0
