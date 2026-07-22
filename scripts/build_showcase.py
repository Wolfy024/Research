"""Train the fixed showcase model and generate README evidence assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from torch.utils.data import Subset

from neural_watermark.data import ImageFolderDataset
from neural_watermark.losses import LossWeights
from neural_watermark.metrics import (
    global_ssim,
    psnr,
)
from neural_watermark.models import ModelConfig, NeuralWatermarker
from neural_watermark.training import (
    TrainConfig,
    evaluate,
    evidence_record,
    make_loader,
    random_bits,
    save_checkpoint,
    seed_everything,
    sha256_file,
    train_steps,
    write_json,
)

BLUE = "#2563EB"
GOLD = "#D4A72C"
INK = "#172033"
MUTED = "#667085"
GRID = "#E5E7EB"
BACKGROUND = "#FAFBFC"


def file_digest(path: Path) -> str:
    return sha256_file(path)


def choose_training_subset(
    dataset: ImageFolderDataset, count: int, seed: int
) -> Subset:
    if count > len(dataset):
        raise ValueError(f"requested {count} training images, found {len(dataset)}")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return Subset(dataset, indices[:count])


@torch.inference_mode()
def per_image_records(
    model: NeuralWatermarker,
    dataset: ImageFolderDataset,
    device: torch.device,
    seed: int,
) -> list[dict[str, object]]:
    model.eval()
    wrong = NeuralWatermarker(
        replace(model.config, key=model.config.key + "::wrong-key")
    )
    wrong.load_state_dict(model.state_dict())
    wrong.to(device).eval()
    loader = make_loader(dataset, 16, False, seed, num_workers=8)
    generator = torch.Generator().manual_seed(seed)
    records: list[dict[str, object]] = []
    offset = 0
    for covers in loader:
        covers = covers.to(device)
        bits = random_bits(
            covers.shape[0],
            model.config.message_length,
            device,
            generator,
        )
        outputs = model(covers, bits)
        wrong_logits = wrong.reveal_details(outputs["watermarked"])["logits"]
        image_psnr = psnr(covers, outputs["watermarked"])
        image_ssim = global_ssim(covers, outputs["watermarked"])
        predicted = outputs["logits"] >= 0
        wrong_predicted = wrong_logits >= 0
        for index in range(covers.shape[0]):
            bit_acc = (predicted[index] == bits[index].bool()).float().mean()
            wrong_acc = (wrong_predicted[index] == bits[index].bool()).float().mean()
            exact = (predicted[index] == bits[index].bool()).all()
            path = dataset.paths[offset + index]
            records.append(
                {
                    "file": path.name,
                    "sha256": file_digest(path),
                    "psnr": float(image_psnr[index]),
                    "ssim": float(image_ssim[index]),
                    "bit_accuracy": float(bit_acc),
                    "exact_message": bool(exact),
                    "wrong_key_bit_accuracy": float(wrong_acc),
                }
            )
        offset += covers.shape[0]
    return records


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(BACKGROUND)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)
    axis.tick_params(colors=MUTED)
    axis.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)


def render_key_chart(metrics: dict[str, object], output: Path) -> None:
    labels = ["Correct key", "Wrong key"]
    values = [
        float(metrics["bit_accuracy"]) * 100,
        float(metrics["wrong_key_bit_accuracy"]) * 100,
    ]
    figure, axis = plt.subplots(figsize=(9.6, 4.8), facecolor=BACKGROUND)
    style_axis(axis)
    bars = axis.barh(
        labels,
        values,
        color=[BLUE, GOLD],
        edgecolor=INK,
        linewidth=0.7,
        height=0.52,
        zorder=2,
    )
    axis.axvline(50, color=INK, linestyle="--", linewidth=1.2)
    axis.text(
        51,
        0.50,
        "Chance · 50%",
        color=INK,
        fontsize=10,
        va="center",
    )
    axis.set_xlim(0, 100)
    axis.set_xlabel("Recovered payload bits (%)", color=INK)
    axis.set_title(
        "Clean-channel recovery by decoder key",
        loc="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
        pad=18,
    )
    axis.text(
        0,
        1.02,
        "32-bit payload · 100 DIV2K validation images · 8-bit PNG quantization",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=10,
    )
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            min(value + 1.5, 95),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            color=INK,
            fontweight="bold",
        )
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_fidelity_chart(records: list[dict[str, object]], output: Path) -> None:
    values = np.array([float(record["psnr"]) for record in records])
    figure, axis = plt.subplots(figsize=(9.6, 4.8), facecolor=BACKGROUND)
    axis.set_facecolor(BACKGROUND)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)
    axis.tick_params(colors=MUTED)
    axis.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    axis.hist(
        values,
        bins=14,
        color=BLUE,
        alpha=0.86,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )
    median = float(np.median(values))
    axis.axvline(median, color=GOLD, linewidth=2, label=f"Median {median:.2f} dB")
    axis.set_xlabel("Cover → watermarked PSNR (dB)", color=INK)
    axis.set_ylabel("Images", color=INK)
    axis.set_title(
        "Visual-fidelity distribution",
        loc="left",
        fontsize=16,
        fontweight="bold",
        color=INK,
        pad=18,
    )
    axis.text(
        0,
        1.02,
        "One observation per DIV2K validation image (n=100)",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=10,
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


@torch.inference_mode()
def render_qualitative(
    model: NeuralWatermarker,
    dataset: ImageFolderDataset,
    output: Path,
    device: torch.device,
    dpi: int = 100,
) -> None:
    model.eval()
    indices = [0, len(dataset) // 2, len(dataset) - 1]
    covers = torch.stack([dataset[index] for index in indices]).to(device)
    generator = torch.Generator().manual_seed(2407)
    bits = random_bits(
        len(indices),
        model.config.message_length,
        device,
        generator,
    )
    outputs = model(covers, bits)
    marked = outputs["watermarked"]
    decoded = outputs["logits"] >= 0

    figure, axes = plt.subplots(
        3,
        3,
        figsize=(10, 10),
        facecolor=BACKGROUND,
    )
    column_titles = ["Cover", "Watermarked", "Residual ×12"]
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=14, fontweight="bold", color=INK)
    for row in range(3):
        residual = (marked[row] - covers[row]) * 12 + 0.5
        panels = [covers[row], marked[row], residual.clamp(0, 1)]
        accuracy = (decoded[row] == bits[row].bool()).float().mean().mul(100).item()
        fidelity = psnr(covers[row : row + 1], marked[row : row + 1]).item()
        for column, panel in enumerate(panels):
            axes[row, column].imshow(panel.permute(1, 2, 0).cpu())
            axes[row, column].axis("off")
        axes[row, 0].set_ylabel(
            f"{fidelity:.2f} dB\n{accuracy:.1f}% bits",
            color=INK,
            fontsize=10,
            rotation=0,
            labelpad=42,
            va="center",
        )
    figure.suptitle(
        "Actual clean-channel model outputs",
        x=0.08,
        y=0.99,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.08,
        0.01,
        "Fixed, unselected DIV2K validation examples · residual centered at gray",
        color=MUTED,
        fontsize=10,
    )
    figure.tight_layout(rect=(0.05, 0.03, 1, 0.96))
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def payload_hex(values: torch.Tensor) -> str:
    bit_string = "".join(
        str(int(value)) for value in values.detach().to(torch.int8).cpu().tolist()
    )
    return f"{int(bit_string, 2):08x}"


def render_bit_grid(
    axis: plt.Axes,
    values: torch.Tensor,
    *,
    target: torch.Tensor | None = None,
    border_color: str = GRID,
) -> None:
    grid = values.detach().to(torch.int8).cpu().numpy().reshape(4, 8)
    expected = None
    if target is not None:
        expected = target.detach().to(torch.int8).cpu().numpy().reshape(4, 8)

    axis.imshow(
        grid,
        cmap=ListedColormap(["#E8EDF5", BLUE]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    for row in range(4):
        for column in range(8):
            value = int(grid[row, column])
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value else INK,
                fontsize=9,
                fontweight="bold",
            )
            if expected is not None and value != int(expected[row, column]):
                axis.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor=GOLD,
                        linewidth=2.4,
                    )
                )
                axis.text(
                    column + 0.29,
                    row - 0.27,
                    "×",
                    ha="center",
                    va="center",
                    color=GOLD,
                    fontsize=8,
                    fontweight="bold",
                )
    axis.add_patch(
        Rectangle(
            (-0.5, -0.5),
            8,
            4,
            fill=False,
            edgecolor=border_color,
            linewidth=2.0,
        )
    )
    for spine in axis.spines.values():
        spine.set_visible(False)


@torch.inference_mode()
def render_key_recovery_examples(
    model: NeuralWatermarker,
    dataset: ImageFolderDataset,
    metrics: dict[str, object],
    output: Path,
    device: torch.device,
    dpi: int = 180,
) -> list[dict[str, object]]:
    """Render actual target, matching-key, and wrong-key payload recovery."""
    model.eval()
    wrong_key_model = NeuralWatermarker(
        replace(model.config, key=model.config.key + "::wrong-key")
    ).to(device)
    wrong_key_model.load_state_dict(model.state_dict())
    wrong_key_model.eval()

    indices = [0, len(dataset) // 2, len(dataset) - 1]
    generator = torch.Generator().manual_seed(2407)
    cohort_bits = random_bits(
        len(dataset),
        model.config.message_length,
        device,
        generator,
    )
    bits = cohort_bits[indices]
    covers = torch.stack([dataset[index] for index in indices]).to(device)
    outputs = model(covers, bits)
    marked = outputs["watermarked"]
    correct = outputs["logits"] >= 0
    wrong = wrong_key_model.reveal_details(marked)["logits"] >= 0

    figure, axes = plt.subplots(
        3,
        4,
        figsize=(14.8, 8.4),
        facecolor=BACKGROUND,
        gridspec_kw={"width_ratios": [1.08, 1, 1, 1]},
    )
    titles = [
        "Watermarked PNG",
        "Embedded ID",
        "Matching-key recovery",
        "Wrong-key recovery",
    ]
    for column, title in enumerate(titles):
        axes[0, column].set_title(
            title,
            fontsize=12,
            fontweight="bold",
            color=INK,
            pad=16,
        )

    example_records: list[dict[str, object]] = []
    for row, index in enumerate(indices):
        axes[row, 0].imshow(marked[row].permute(1, 2, 0).cpu())
        axes[row, 0].axis("off")
        render_bit_grid(axes[row, 1], bits[row], border_color=INK)
        render_bit_grid(
            axes[row, 2],
            correct[row],
            target=bits[row],
            border_color=BLUE,
        )
        render_bit_grid(
            axes[row, 3],
            wrong[row],
            target=bits[row],
            border_color=GOLD,
        )
        axes[row, 1].set_xlabel(
            f"0x{payload_hex(bits[row])}",
            color=INK,
            fontsize=9,
            fontfamily="monospace",
            labelpad=5,
        )
        axes[row, 2].set_xlabel(
            f"0x{payload_hex(correct[row])}",
            color=BLUE,
            fontsize=9,
            fontfamily="monospace",
            labelpad=5,
        )
        axes[row, 3].set_xlabel(
            f"0x{payload_hex(wrong[row])}",
            color=INK,
            fontsize=9,
            fontfamily="monospace",
            labelpad=5,
        )

        fidelity = psnr(covers[row : row + 1], marked[row : row + 1]).item()
        similarity = global_ssim(covers[row : row + 1], marked[row : row + 1]).item()
        correct_accuracy = (correct[row] == bits[row].bool()).float().mean().item()
        wrong_accuracy = (wrong[row] == bits[row].bool()).float().mean().item()
        axes[row, 0].text(
            0.5,
            -0.06,
            f"{dataset.paths[index].name} · {fidelity:.2f} dB · SSIM {similarity:.6f}",
            transform=axes[row, 0].transAxes,
            color=INK,
            fontsize=8,
            ha="center",
            va="top",
        )
        example_records.append(
            {
                "dataset_index": index,
                "file": dataset.paths[index].name,
                "sha256": file_digest(dataset.paths[index]),
                "psnr": fidelity,
                "ssim": similarity,
                "target_hex": payload_hex(bits[row]),
                "matching_key_hex": payload_hex(correct[row]),
                "wrong_key_hex": payload_hex(wrong[row]),
                "matching_key_bit_accuracy": correct_accuracy,
                "wrong_key_bit_accuracy": wrong_accuracy,
            }
        )

    exact_messages = round(
        float(metrics["exact_message_accuracy"]) * int(metrics["images"])
    )
    wrong_exact_messages = round(
        float(metrics["wrong_key_exact_message_accuracy"]) * int(metrics["images"])
    )
    figure.suptitle(
        "Blind payload recovery from the same watermarked PNG",
        x=0.06,
        y=0.985,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.06,
        0.02,
        f"Matching key: {float(metrics['bit_accuracy']) * 100:.2f}% bits · "
        f"{exact_messages}/{int(metrics['images'])} exact IDs   |   "
        f"Wrong key: {float(metrics['wrong_key_bit_accuracy']) * 100:.2f}% bits · "
        f"{wrong_exact_messages}/{int(metrics['images'])} exact IDs   |   "
        "gold × = mismatch",
        color=INK,
        fontsize=10,
        fontweight="bold",
    )
    figure.text(
        0.06,
        0.002,
        "Fixed, unselected DIV2K examples · 32-bit IDs shown row-major · "
        "neither decoder receives the cover",
        color=MUTED,
        fontsize=9,
    )
    figure.tight_layout(rect=(0.07, 0.065, 1, 0.90), h_pad=1.8, w_pad=1.4)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return example_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--train-images", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--asset-dir", default="assets")
    parser.add_argument("--checkpoint", default="checkpoints/showcase.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    seed = 24
    seed_everything(seed)
    output_dir = Path(args.output_dir)
    asset_dir = Path(args.asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    model_config = ModelConfig(
        image_size=256,
        message_length=32,
        key="wolfy024-provenance-v1",
        hidden_channels=32,
        max_coefficient_delta=0.12,
        quantize=True,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=3e-4,
        weight_decay=1e-5,
        max_steps=args.steps,
        seed=seed,
        grad_clip=1.0,
        num_workers=8,
    )
    train_dataset_full = ImageFolderDataset(args.train_dir, model_config.image_size)
    train_dataset = choose_training_subset(
        train_dataset_full,
        args.train_images,
        seed,
    )
    eval_dataset = ImageFolderDataset(args.eval_dir, model_config.image_size)

    model = NeuralWatermarker(model_config).to(device)
    baseline = evaluate(
        model,
        make_loader(eval_dataset, 8, False, 2407, num_workers=8),
        device,
        seed=2407,
    )
    train_metrics = train_steps(
        model,
        make_loader(
            train_dataset,
            train_config.batch_size,
            True,
            seed,
            num_workers=train_config.num_workers,
        ),
        train_config,
        LossWeights(target_psnr=40.0),
        device,
    )
    checkpoint = save_checkpoint(args.checkpoint, model, train_config, train_metrics)
    final = evaluate(
        model,
        make_loader(eval_dataset, 8, False, 2407, num_workers=8),
        device,
        seed=2407,
    )
    records = per_image_records(model, eval_dataset, device, seed=2407)

    manifest = {
        "schema_version": 1,
        "training": {
            "dataset": "COCO val2017",
            "selection": "seeded shuffle, first cohort",
            "seed": seed,
            "images": len(train_dataset),
            "source_files_sha256": hashlib.sha256(
                "\n".join(
                    train_dataset_full.paths[index].name
                    for index in train_dataset.indices
                ).encode()
            ).hexdigest(),
        },
        "evaluation": {
            "dataset": "DIV2K validation",
            "images": len(eval_dataset),
            "files": [
                {"file": path.name, "sha256": file_digest(path)}
                for path in eval_dataset.paths
            ],
        },
    }
    manifest_path = write_json(output_dir / "showcase_manifest.json", manifest)
    record = evidence_record(
        "cross_dataset_clean_channel_showcase",
        model,
        {
            "untrained_baseline": baseline,
            "training": train_metrics,
            "final": final,
            "per_image": records,
        },
        {
            "training": "COCO val2017 deterministic 2,000-image subset",
            "evaluation": "DIV2K validation, all 100 images",
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
        checkpoint,
    )
    record["chart_contracts"] = {
        "key_separation": {
            "question": "Does the matching key recover more bits than a wrong key?",
            "chart": "two-category horizontal bar with chance reference",
            "source": "metrics.final",
        },
        "fidelity_distribution": {
            "question": "How does visual fidelity vary across the holdout?",
            "chart": "histogram with median reference",
            "source": "metrics.per_image[].psnr",
        },
        "key_recovery_examples": {
            "question": (
                "What payload does each decoder key recover from the same PNG?"
            ),
            "chart": "fixed examples with watermarked image and 32-bit payload grids",
            "source": "recovery_examples and metrics.final",
        },
    }
    render_key_chart(final, asset_dir / "key_separation.png")
    render_fidelity_chart(records, asset_dir / "fidelity_distribution.png")
    render_qualitative(
        model, eval_dataset, asset_dir / "qualitative_examples.png", device
    )
    record["recovery_examples"] = render_key_recovery_examples(
        model,
        eval_dataset,
        final,
        asset_dir / "key_recovery_examples.png",
        device,
    )
    write_json(output_dir / "showcase.json", record)

    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "baseline": baseline,
        "final": final,
        "figures": sorted(str(path) for path in asset_dir.glob("*.png")),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
