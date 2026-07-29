from streamer import RTMPStreamer


def test_nvenc_command_uses_low_latency_options():
    command = RTMPStreamer("rtmp://example/live", encoder="h264_nvenc", preset="p4")._command()

    assert command[command.index("-c:v") + 1] == "h264_nvenc"
    assert command[command.index("-preset") + 1] == "p4"
    assert command[command.index("-tune") + 1] == "ll"
    assert "-threads" not in command


def test_libx264_command_has_valid_thread_and_tune_arguments():
    command = RTMPStreamer("rtmp://example/live", encoder="libx264", preset="veryfast")._command()

    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-threads") + 1] == "4"
    assert command[command.index("-preset") + 1] == "veryfast"
    assert command[command.index("-tune") + 1] == "zerolatency"
