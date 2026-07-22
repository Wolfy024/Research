"""Deterministic image loading and a procedural smoke-test dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def discover_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {root}")
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"no supported images found under {root}")
    return paths


def load_image(path: str | Path, image_size: int) -> Tensor:
    with Image.open(path) as source:
        image = ImageOps.fit(
            source.convert("RGB"),
            (image_size, image_size),
            method=Image.Resampling.LANCZOS,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def save_png(image: Tensor, path: str | Path) -> None:
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("save_png accepts only one image")
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape [3, H, W]")
    array = (
        image.detach()
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    destination = Path(path)
    if destination.suffix.lower() != ".png":
        raise ValueError("the supported lossless output format is .png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(destination, format="PNG")


class ImageFolderDataset(Dataset[Tensor]):
    def __init__(self, root: str | Path, image_size: int) -> None:
        self.paths = discover_images(root)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        return load_image(self.paths[index], self.image_size)


class SyntheticTextureDataset(Dataset[Tensor]):
    """Procedural covers for tests only; never use these for headline results."""

    def __init__(self, length: int, image_size: int, seed: int = 0) -> None:
        if length < 1:
            raise ValueError("length must be positive")
        self.length = length
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tensor:
        generator = torch.Generator().manual_seed(self.seed + index)
        coarse_size = max(4, self.image_size // 8)
        coarse = torch.rand(1, 3, coarse_size, coarse_size, generator=generator)
        smooth = F.interpolate(
            coarse,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        )[0]
        texture = torch.rand(
            1, 3, self.image_size, self.image_size, generator=generator
        )
        texture = F.avg_pool2d(texture, kernel_size=3, stride=1, padding=1)[0]

        axis = torch.linspace(0, 1, self.image_size)
        vertical, horizontal = torch.meshgrid(axis, axis, indexing="ij")
        gradients = torch.stack([horizontal, vertical, (horizontal + vertical) / 2])
        mix = torch.rand(3, 1, 1, generator=generator)
        return (0.65 * smooth + 0.2 * texture + 0.15 * gradients * mix).clamp(0, 1)
