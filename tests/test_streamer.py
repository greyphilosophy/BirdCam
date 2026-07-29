from streamer import RTMPStreamer


def test_nvenc_command_uses_low_latency_options():
    command = RTMPStreamer("rtmp://example/live", encoder="h264_nvenc", preset="p4")._command()

    assert command[command.index("-c:v") + 1] == "h264_nvenc"
    assert command[command.index("-preset") + 1] == "p4"
    assert command[command.index("-tune") + 1] == "ll"
    assert "-threads" not in command
    assert "-an" in command


def test_libx264_command_has_valid_thread_and_tune_arguments():
    command = RTMPStreamer("rtmp://example/live", encoder="libx264", preset="veryfast")._command()

    assert command[command.index("-c:v") + 1] == "libx264"
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


def test_safe_command_redacts_rtmp_url():
    streamer = RTMPStreamer("rtmp://example/live/secret-key")
    safe_text = streamer._safe_command_text(streamer._command())

    assert "secret-key" not in safe_text
    assert "<redacted-rtmp-url>" in safe_text