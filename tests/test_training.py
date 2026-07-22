from pathlib import Path

import torch

from neural_watermark.data import SyntheticTextureDataset
from neural_watermark.models import ModelConfig, NeuralWatermarker
from neural_watermark.training import (
    TrainConfig,
    evaluate,
    load_checkpoint,
    make_loader,
    save_checkpoint,
)


def test_checkpoint_round_trip_and_evaluation(tmp_path: Path) -> None:
    config = ModelConfig(
        image_size=16,
        message_length=2,
        hidden_channels=4,
        quantize=True,
    )
    model = NeuralWatermarker(config)
    train_config = TrainConfig(batch_size=2, max_steps=1)
    path = save_checkpoint(tmp_path / "model.pt", model, train_config, {})
    restored, payload = load_checkpoint(path, torch.device("cpu"))

    assert payload["format_version"] == 1
    for original, loaded in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(original, loaded)

    dataset = SyntheticTextureDataset(2, 16)
    loader = make_loader(dataset, 2, False, 1)
    metrics = evaluate(restored, loader, torch.device("cpu"), seed=1)
    assert metrics["images"] == 2
    assert 0 <= metrics["bit_accuracy"] <= 1
    assert 0 <= metrics["wrong_key_bit_accuracy"] <= 1
