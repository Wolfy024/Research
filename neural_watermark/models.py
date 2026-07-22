"""Blind keyed watermarking with block-DCT carriers and learned host cancellation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .carriers import (
    FREQUENCIES,
    KeyedBlockLayout,
    block_dct,
    block_idct,
    orthonormal_dct_matrix,
)


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 256
    message_length: int = 32
    key: str = "research-demo-key"
    hidden_channels: int = 32
    max_coefficient_delta: float = 0.12
    quantize: bool = True

    def __post_init__(self) -> None:
        if self.image_size % 8:
            raise ValueError("image_size must be divisible by 8")
        if self.message_length > (self.image_size // 8) ** 2:
            raise ValueError("message_length cannot exceed the number of 8x8 blocks")
        if self.max_coefficient_delta <= 0:
            raise ValueError("max_coefficient_delta must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def rgb_luminance(image: Tensor) -> Tensor:
    weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def quantize_8bit_straight_through(image: Tensor) -> Tensor:
    quantized = torch.round(image * 255.0) / 255.0
    return image + (quantized - image).detach()


class TextureGainNetwork(nn.Module):
    """Predict one bounded embedding gain per 8x8 image block."""

    def __init__(self, hidden_channels: int, max_delta: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=8, stride=8),
        )
        nn.init.zeros_(self.features[-1].weight)
        nn.init.zeros_(self.features[-1].bias)
        self.max_delta = max_delta

    def forward(self, cover: Tensor) -> Tensor:
        logits = self.features(cover).flatten(1)
        return self.max_delta * (0.25 + 0.75 * torch.sigmoid(logits))


class HostPredictor(nn.Module):
    """Estimate a block's unmarked carrier coefficient from its other coefficients."""

    def __init__(self, hidden_channels: int, embedding_dim: int = 8) -> None:
        super().__init__()
        self.frequency_embedding = nn.Embedding(len(FREQUENCIES), embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(64 + embedding_dim, hidden_channels * 2),
            nn.SiLU(),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, context: Tensor, frequency_indices: Tensor) -> Tensor:
        batch, blocks, _ = context.shape
        embeddings = self.frequency_embedding(frequency_indices)
        embeddings = embeddings.unsqueeze(0).expand(batch, -1, -1)
        features = torch.cat([context / 4.0, embeddings], dim=-1)
        return self.network(features.reshape(batch * blocks, -1)).reshape(batch, blocks)


class NeuralWatermarker(nn.Module):
    """Keyed provenance watermark with content-adaptive DCT coefficient gains."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.layout = KeyedBlockLayout(
            self.config.message_length,
            self.config.image_size,
            self.config.key,
        )
        self.register_buffer("dct_matrix", orthonormal_dct_matrix(), persistent=False)
        self.gain_network = TextureGainNetwork(
            self.config.hidden_channels,
            self.config.max_coefficient_delta,
        )
        self.host_predictor = HostPredictor(self.config.hidden_channels)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0)))
        self.bit_bias = nn.Parameter(torch.zeros(self.config.message_length))

    def _check_images(self, images: Tensor) -> None:
        expected = (3, self.config.image_size, self.config.image_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(
                f"expected images [B, {expected}], got {tuple(images.shape)}"
            )

    def _coefficients(self, image: Tensor) -> Tensor:
        return block_dct(rgb_luminance(image), self.dct_matrix)

    def embed(self, cover: Tensor, bits: Tensor) -> dict[str, Tensor]:
        self._check_images(cover)
        if bits.ndim != 2 or bits.shape[1] != self.config.message_length:
            raise ValueError(
                f"expected bits [B, {self.config.message_length}], "
                f"got {tuple(bits.shape)}"
            )
        if bits.min().item() < 0 or bits.max().item() > 1:
            raise ValueError("bits must be in [0, 1]")

        cover_coefficients = self._coefficients(cover)
        gains = self.gain_network(cover)
        bipolar_by_block = self.layout.message_by_block(bits.mul(2).sub(1))
        coefficient_values = bipolar_by_block * self.layout.polarities[None] * gains
        coefficient_delta = self.layout.coefficient_delta(coefficient_values)
        luminance_residual = block_idct(
            coefficient_delta,
            self.dct_matrix,
            self.layout.grid_size,
        )
        pre_quantized = (cover + luminance_residual.expand(-1, 3, -1, -1)).clamp(
            0.0, 1.0
        )
        watermarked = (
            quantize_8bit_straight_through(pre_quantized)
            if self.config.quantize
            else pre_quantized
        )
        return {
            "watermarked": watermarked,
            "pre_quantized": pre_quantized,
            "residual": watermarked - cover,
            "gains": gains,
            "host_target": self.layout.select(cover_coefficients),
        }

    def reveal_details(self, watermarked: Tensor) -> dict[str, Tensor]:
        self._check_images(watermarked)
        coefficients = self._coefficients(watermarked)
        observed = self.layout.select(coefficients)
        context = self.layout.remove_selected(coefficients)
        host_prediction = self.host_predictor(
            context,
            self.layout.frequency_indices,
        )
        block_evidence = (observed - host_prediction) * self.layout.polarities[None]
        bit_evidence = self.layout.aggregate(block_evidence)
        logits = bit_evidence * self.logit_scale.exp().clamp(max=500.0) + self.bit_bias
        return {
            "logits": logits,
            "host_prediction": host_prediction,
            "block_evidence": block_evidence,
            "bit_evidence": bit_evidence,
        }

    def forward(self, cover: Tensor, bits: Tensor) -> dict[str, Tensor]:
        outputs = self.embed(cover, bits)
        outputs.update(self.reveal_details(outputs["watermarked"]))
        return outputs

    def reveal(self, watermarked: Tensor) -> Tensor:
        return torch.sigmoid(self.reveal_details(watermarked)["logits"])

    def decode(self, watermarked: Tensor) -> Tensor:
        return (self.reveal(watermarked) >= 0.5).to(watermarked.dtype)
