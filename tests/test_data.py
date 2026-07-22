from pathlib import Path

import torch

from neural_watermark.data import (
    ImageFolderDataset,
    SyntheticTextureDataset,
    discover_images,
    save_png,
)


def test_synthetic_dataset_is_deterministic() -> None:
    dataset = SyntheticTextureDataset(3, 16, seed=7)
    torch.testing.assert_close(dataset[1], dataset[1])
    assert not torch.equal(dataset[0], dataset[1])


def test_png_round_trip_and_discovery(tmp_path: Path) -> None:
    image = torch.rand(3, 16, 16)
    path = tmp_path / "nested" / "cover.png"
    save_png(image, path)
    assert discover_images(tmp_path) == [path]

    dataset = ImageFolderDataset(tmp_path, 16)
    expected = image.mul(255).round().div(255)
    torch.testing.assert_close(dataset[0], expected)
