import torch

from neural_watermark.cli import bits_to_hex, main, message_to_bits


def test_message_hex_round_trip() -> None:
    bits = message_to_bits("a5", 8, torch.device("cpu"))
    assert bits_to_hex(bits) == "a5"


def test_smoke_command_writes_artifacts(tmp_path) -> None:
    checkpoint = tmp_path / "smoke.pt"
    evidence = tmp_path / "smoke.json"
    result = main(
        [
            "smoke",
            "--steps",
            "1",
            "--samples",
            "2",
            "--batch-size",
            "2",
            "--image-size",
            "16",
            "--message-length",
            "2",
            "--hidden-channels",
            "4",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(evidence),
        ]
    )
    assert result == 0
    assert checkpoint.is_file()
    assert evidence.is_file()
