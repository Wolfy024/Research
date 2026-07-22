"""Dependency-free clean-channel watermark evaluation metrics."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def psnr(reference: Tensor, candidate: Tensor, data_range: float = 1.0) -> Tensor:
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have the same shape")
    mse = (reference - candidate).square().flatten(1).mean(dim=1)
    peak = torch.tensor(data_range * data_range, device=mse.device, dtype=mse.dtype)
    return torch.where(
        mse == 0,
        torch.full_like(mse, float("inf")),
        10.0 * torch.log10(peak / mse),
    )


def global_ssim(
    reference: Tensor,
    candidate: Tensor,
    data_range: float = 1.0,
) -> Tensor:
    """Compute global SSIM per image across channels and spatial dimensions."""
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have the same shape")
    dimensions = tuple(range(1, reference.ndim))
    reference_mean = reference.mean(dim=dimensions)
    candidate_mean = candidate.mean(dim=dimensions)
    reference_centered = reference - reference_mean.view(-1, 1, 1, 1)
    candidate_centered = candidate - candidate_mean.view(-1, 1, 1, 1)
    reference_variance = reference_centered.square().mean(dim=dimensions)
    candidate_variance = candidate_centered.square().mean(dim=dimensions)
    covariance = (reference_centered * candidate_centered).mean(dim=dimensions)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * reference_mean * candidate_mean + c1) * (2 * covariance + c2)
    denominator = (reference_mean.square() + candidate_mean.square() + c1) * (
        reference_variance + candidate_variance + c2
    )
    return numerator / denominator


def _prediction_threshold(values: Tensor) -> float:
    return 0.0 if values.min().item() < 0 or values.max().item() > 1 else 0.5


def bit_accuracy(logits_or_probabilities: Tensor, bits: Tensor) -> Tensor:
    if logits_or_probabilities.shape != bits.shape:
        raise ValueError("predictions and bits must have the same shape")
    if logits_or_probabilities.numel() == 0:
        raise ValueError("predictions cannot be empty")
    predicted = logits_or_probabilities >= _prediction_threshold(
        logits_or_probabilities
    )
    return (predicted == bits.bool()).float().mean()


def exact_message_accuracy(logits_or_probabilities: Tensor, bits: Tensor) -> Tensor:
    if logits_or_probabilities.shape != bits.shape:
        raise ValueError("predictions and bits must have the same shape")
    predicted = logits_or_probabilities >= _prediction_threshold(
        logits_or_probabilities
    )
    return (predicted == bits.bool()).all(dim=1).float().mean()


def bit_error_rate(logits_or_probabilities: Tensor, bits: Tensor) -> Tensor:
    return 1.0 - bit_accuracy(logits_or_probabilities, bits)


def residual_rms(reference: Tensor, candidate: Tensor) -> Tensor:
    return (reference - candidate).square().mean().sqrt()


def theoretical_psnr_for_rms(rms: float) -> float:
    if rms < 0:
        raise ValueError("rms cannot be negative")
    return math.inf if rms == 0 else -20.0 * math.log10(rms)
