"""Objectives for clean, lossless-channel provenance watermarking."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class LossWeights:
    bit: float = 1.0
    fidelity_budget: float = 5000.0
    host_prediction: float = 1.0
    low_frequency: float = 100.0
    gain_smoothness: float = 0.02
    target_psnr: float = 40.0


def grid_total_variation(values: Tensor, grid_size: int) -> Tensor:
    grid = values.reshape(values.shape[0], 1, grid_size, grid_size)
    dx = grid[..., :, 1:] - grid[..., :, :-1]
    dy = grid[..., 1:, :] - grid[..., :-1, :]
    return dx.abs().mean() + dy.abs().mean()


class WatermarkObjective(nn.Module):
    """Balance exact payload recovery with a configurable PSNR budget."""

    def __init__(self, grid_size: int, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.weights = weights or LossWeights()

    def forward(
        self,
        cover: Tensor,
        bits: Tensor,
        outputs: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        bit_loss = F.binary_cross_entropy_with_logits(outputs["logits"], bits)
        image_mse = F.mse_loss(outputs["watermarked"], cover)
        budget = 10.0 ** (-self.weights.target_psnr / 10.0)
        fidelity_excess = torch.relu(image_mse - budget)
        host_loss = F.smooth_l1_loss(
            outputs["host_prediction"],
            outputs["host_target"],
        )
        residual = outputs["watermarked"] - cover
        low_frequency = (
            F.avg_pool2d(residual, kernel_size=9, stride=1, padding=4).square().mean()
        )
        gain_smoothness = grid_total_variation(outputs["gains"], self.grid_size)

        total = (
            self.weights.bit * bit_loss
            + self.weights.fidelity_budget * fidelity_excess
            + self.weights.host_prediction * host_loss
            + self.weights.low_frequency * low_frequency
            + self.weights.gain_smoothness * gain_smoothness
        )
        components = {
            "total": total.detach(),
            "bit": bit_loss.detach(),
            "image_mse": image_mse.detach(),
            "fidelity_excess": fidelity_excess.detach(),
            "host_prediction": host_loss.detach(),
            "low_frequency": low_frequency.detach(),
            "gain_smoothness": gain_smoothness.detach(),
            "target_mse": torch.tensor(budget, device=cover.device, dtype=cover.dtype),
        }
        return total, components
