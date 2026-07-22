"""Training, evaluation, checkpoint, and evidence utilities."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .losses import LossWeights, WatermarkObjective
from .metrics import (
    bit_accuracy,
    bit_error_rate,
    exact_message_accuracy,
    global_ssim,
    psnr,
    residual_rms,
)
from .models import ModelConfig, NeuralWatermarker


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_steps: int = 2000
    seed: int = 24
    grad_clip: float = 1.0
    num_workers: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def make_loader(
    dataset: torch.utils.data.Dataset[Tensor],
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader[Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def random_bits(
    batch_size: int,
    message_length: int,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    return torch.randint(
        0,
        2,
        (batch_size, message_length),
        generator=generator,
        dtype=torch.float32,
    ).to(device)


def train_steps(
    model: NeuralWatermarker,
    loader: DataLoader[Tensor],
    train_config: TrainConfig,
    loss_weights: LossWeights,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    objective = WatermarkObjective(model.layout.grid_size, loss_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    bit_generator = torch.Generator().manual_seed(train_config.seed + 1009)
    iterator: Iterable[Tensor] = iter(loader)
    totals: dict[str, float] = {}
    final_accuracy = 0.0

    for _step in range(train_config.max_steps):
        try:
            covers = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            covers = next(iterator)
        covers = covers.to(device)
        bits = random_bits(
            covers.shape[0],
            model.config.message_length,
            device,
            bit_generator,
        )

        optimizer.zero_grad(set_to_none=True)
        outputs = model(covers, bits)
        loss, components = objective(covers, bits, outputs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()

        for name, value in components.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        final_accuracy = float(bit_accuracy(outputs["logits"].detach(), bits))

    metrics = {
        f"mean_{name}": value / train_config.max_steps for name, value in totals.items()
    }
    metrics["final_batch_bit_accuracy"] = final_accuracy
    metrics["steps"] = float(train_config.max_steps)
    return metrics


@torch.inference_mode()
def evaluate(
    model: NeuralWatermarker,
    loader: DataLoader[Tensor],
    device: torch.device,
    seed: int,
    max_batches: int | None = None,
    include_wrong_key: bool = True,
) -> dict[str, float | int]:
    model.eval()
    bit_generator = torch.Generator().manual_seed(seed)
    wrong_key_model = None
    if include_wrong_key:
        wrong_config = replace(model.config, key=model.config.key + "::wrong-key")
        wrong_key_model = NeuralWatermarker(wrong_config).to(device)
        wrong_key_model.load_state_dict(model.state_dict())
        wrong_key_model.eval()

    totals = {
        "psnr": 0.0,
        "ssim": 0.0,
        "bit_accuracy": 0.0,
        "bit_error_rate": 0.0,
        "exact_message_accuracy": 0.0,
        "residual_rms": 0.0,
        "wrong_key_bit_accuracy": 0.0,
        "wrong_key_exact_message_accuracy": 0.0,
    }
    images = 0
    batches = 0
    for covers in loader:
        if max_batches is not None and batches >= max_batches:
            break
        covers = covers.to(device)
        bits = random_bits(
            covers.shape[0],
            model.config.message_length,
            device,
            bit_generator,
        )
        outputs = model(covers, bits)
        batch_size = covers.shape[0]
        totals["psnr"] += float(psnr(covers, outputs["watermarked"]).sum())
        totals["ssim"] += float(global_ssim(covers, outputs["watermarked"]).sum())
        totals["bit_accuracy"] += (
            float(bit_accuracy(outputs["logits"], bits)) * batch_size
        )
        totals["bit_error_rate"] += (
            float(bit_error_rate(outputs["logits"], bits)) * batch_size
        )
        totals["exact_message_accuracy"] += (
            float(exact_message_accuracy(outputs["logits"], bits)) * batch_size
        )
        totals["residual_rms"] += (
            float(residual_rms(covers, outputs["watermarked"])) * batch_size
        )

        if wrong_key_model is not None:
            wrong_logits = wrong_key_model.reveal_details(outputs["watermarked"])[
                "logits"
            ]
            totals["wrong_key_bit_accuracy"] += (
                float(bit_accuracy(wrong_logits, bits)) * batch_size
            )
            totals["wrong_key_exact_message_accuracy"] += (
                float(exact_message_accuracy(wrong_logits, bits)) * batch_size
            )
        images += batch_size
        batches += 1

    if images == 0:
        raise ValueError("evaluation loader produced no images")
    results: dict[str, float | int] = {
        name: value / images for name, value in totals.items()
    }
    results["images"] = images
    results["batches"] = batches
    results["payload_bits"] = model.config.message_length
    results["bits_per_pixel"] = model.config.message_length / (
        model.config.image_size**2
    )
    return results


def save_checkpoint(
    path: str | Path,
    model: NeuralWatermarker,
    train_config: TrainConfig,
    train_metrics: dict[str, float],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_config": model.config.to_dict(),
            "train_config": train_config.to_dict(),
            "state_dict": model.state_dict(),
            "train_metrics": train_metrics,
        },
        destination,
    )
    return destination


def load_checkpoint(
    path: str | Path,
    device: torch.device,
    key_override: str | None = None,
) -> tuple[NeuralWatermarker, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    config_dict = dict(payload["model_config"])
    if key_override is not None:
        config_dict["key"] = key_override
    model = NeuralWatermarker(ModelConfig(**config_dict)).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def evidence_record(
    kind: str,
    model: NeuralWatermarker,
    metrics: dict[str, object],
    dataset: dict[str, object],
    checkpoint: str | Path | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": 1,
        "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "device": str(next(model.parameters()).device),
        },
        "model": {
            "config": model.config.to_dict(),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "dataset": dataset,
        "metrics": metrics,
        "scope": {
            "channel": "clean lossless PNG / in-memory 8-bit quantization",
            "jpeg": "not supported or evaluated",
            "corruption_attacks": "not supported or evaluated",
        },
    }
    if checkpoint is not None:
        record["checkpoint"] = {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        }
    return record


def write_json(path: str | Path, payload: dict[str, object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return destination
