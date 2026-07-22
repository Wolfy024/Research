"""Keyed neural spread-spectrum image watermarking."""

from .losses import LossWeights, WatermarkObjective
from .metrics import bit_accuracy, bit_error_rate, psnr
from .models import ModelConfig, NeuralWatermarker

__all__ = [
    "LossWeights",
    "ModelConfig",
    "NeuralWatermarker",
    "WatermarkObjective",
    "bit_accuracy",
    "bit_error_rate",
    "psnr",
]

__version__ = "0.1.0"
