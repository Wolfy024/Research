"""Command-line interface for training, embedding, extracting, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import ImageFolderDataset, SyntheticTextureDataset, load_image, save_png
from .losses import LossWeights
from .models import ModelConfig, NeuralWatermarker
from .training import (
    TrainConfig,
    evaluate,
    evidence_record,
    load_checkpoint,
    make_loader,
    resolve_device,
    save_checkpoint,
    seed_everything,
    train_steps,
    write_json,
)


def message_to_bits(message: str, length: int, device: torch.device) -> torch.Tensor:
    value = int(message.removeprefix("0x"), 16)
    if value >= 1 << length:
        raise ValueError(f"message does not fit in {length} bits")
    bit_string = f"{value:0{length}b}"
    return torch.tensor(
        [[float(character) for character in bit_string]],
        device=device,
    )


def bits_to_hex(bits: torch.Tensor) -> str:
    characters = "".join(str(int(value)) for value in bits.flatten().tolist())
    width = (len(characters) + 3) // 4
    return f"{int(characters, 2):0{width}x}"


def load_config(path: str | Path) -> tuple[ModelConfig, TrainConfig, LossWeights]:
    payload = json.loads(Path(path).read_text())
    return (
        ModelConfig(**payload.get("model", {})),
        TrainConfig(**payload.get("training", {})),
        LossWeights(**payload.get("loss", {})),
    )


def run_smoke(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    seed_everything(args.seed)
    model_config = ModelConfig(
        image_size=args.image_size,
        message_length=args.message_length,
        key=args.key,
        hidden_channels=args.hidden_channels,
        quantize=True,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.steps,
        seed=args.seed,
    )
    dataset = SyntheticTextureDataset(args.samples, args.image_size, args.seed)
    loader = make_loader(dataset, args.batch_size, True, args.seed)
    model = NeuralWatermarker(model_config).to(device)
    train_metrics = train_steps(
        model,
        loader,
        train_config,
        LossWeights(target_psnr=args.target_psnr),
        device,
    )
    checkpoint = save_checkpoint(args.checkpoint, model, train_config, train_metrics)
    evaluation_loader = make_loader(dataset, args.batch_size, False, args.seed)
    metrics = evaluate(model, evaluation_loader, device, args.seed, max_batches=4)
    record = evidence_record(
        "synthetic_smoke_test",
        model,
        {"training": train_metrics, "evaluation": metrics},
        {
            "name": "procedural synthetic textures",
            "samples": len(dataset),
            "seed": args.seed,
            "headline_eligible": False,
        },
        checkpoint,
    )
    write_json(args.output, record)
    print(json.dumps(record["metrics"], indent=2, sort_keys=True))
    return 0


def run_train(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    model_config, train_config, loss_weights = load_config(args.config)
    if args.steps is not None:
        train_config = TrainConfig(
            **{**train_config.to_dict(), "max_steps": args.steps}
        )
    seed_everything(train_config.seed)
    dataset = ImageFolderDataset(args.data_dir, model_config.image_size)
    loader = make_loader(
        dataset,
        train_config.batch_size,
        True,
        train_config.seed,
        train_config.num_workers,
    )
    model = NeuralWatermarker(model_config).to(device)
    metrics = train_steps(model, loader, train_config, loss_weights, device)
    checkpoint = save_checkpoint(args.checkpoint, model, train_config, metrics)
    print(json.dumps({"checkpoint": str(checkpoint), "training": metrics}, indent=2))
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device, args.key)
    dataset = ImageFolderDataset(args.data_dir, model.config.image_size)
    loader = make_loader(dataset, args.batch_size, False, args.seed)
    metrics = evaluate(model, loader, device, args.seed, args.max_batches)
    record = evidence_record(
        "clean_channel_evaluation",
        model,
        metrics,
        {
            "name": str(Path(args.data_dir).resolve()),
            "images_discovered": len(dataset),
            "seed": args.seed,
        },
        args.checkpoint,
    )
    write_json(args.output, record)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def run_embed(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device, args.key)
    cover = load_image(args.input, model.config.image_size).unsqueeze(0).to(device)
    bits = message_to_bits(args.message, model.config.message_length, device)
    with torch.inference_mode():
        watermarked = model.embed(cover, bits)["watermarked"]
    save_png(watermarked, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "message_hex": bits_to_hex(bits),
                "payload_bits": model.config.message_length,
                "format": "PNG",
            },
            indent=2,
        )
    )
    return 0


def run_extract(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device, args.key)
    image = load_image(args.input, model.config.image_size).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model.reveal(image)
        bits = (probabilities >= 0.5).to(torch.int64)
    print(
        json.dumps(
            {
                "message_hex": bits_to_hex(bits),
                "payload_bits": model.config.message_length,
                "mean_confidence": float(
                    torch.maximum(probabilities, 1 - probabilities).mean()
                ),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spectral-watermark",
        description="Blind keyed block-DCT provenance watermarking",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="run a synthetic training smoke test")
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--steps", type=int, default=8)
    smoke.add_argument("--samples", type=int, default=24)
    smoke.add_argument("--batch-size", type=int, default=4)
    smoke.add_argument("--image-size", type=int, default=32)
    smoke.add_argument("--message-length", type=int, default=8)
    smoke.add_argument("--hidden-channels", type=int, default=8)
    smoke.add_argument("--learning-rate", type=float, default=1e-3)
    smoke.add_argument("--target-psnr", type=float, default=40.0)
    smoke.add_argument("--key", default="smoke-test-key")
    smoke.add_argument("--seed", type=int, default=24)
    smoke.add_argument("--checkpoint", default="runs/smoke/latest.pt")
    smoke.add_argument("--output", default="results/smoke.json")
    smoke.set_defaults(handler=run_smoke)

    train = subparsers.add_parser("train", help="train on a directory of images")
    train.add_argument("--config", default="configs/default.json")
    train.add_argument("--data-dir", required=True)
    train.add_argument("--checkpoint", default="runs/default/latest.pt")
    train.add_argument("--steps", type=int)
    train.add_argument("--device", default="auto")
    train.set_defaults(handler=run_train)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate a checkpoint on a clean image directory"
    )
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--data-dir", required=True)
    evaluate_parser.add_argument("--output", default="results/evaluation.json")
    evaluate_parser.add_argument("--key")
    evaluate_parser.add_argument("--batch-size", type=int, default=16)
    evaluate_parser.add_argument("--max-batches", type=int)
    evaluate_parser.add_argument("--seed", type=int, default=2407)
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.set_defaults(handler=run_evaluate)

    embed = subparsers.add_parser("embed", help="embed a hex provenance message")
    embed.add_argument("--checkpoint", required=True)
    embed.add_argument("--input", required=True)
    embed.add_argument("--output", required=True)
    embed.add_argument("--message", required=True)
    embed.add_argument("--key")
    embed.add_argument("--device", default="auto")
    embed.set_defaults(handler=run_embed)

    extract = subparsers.add_parser("extract", help="extract a hex provenance message")
    extract.add_argument("--checkpoint", required=True)
    extract.add_argument("--input", required=True)
    extract.add_argument("--key")
    extract.add_argument("--device", default="auto")
    extract.set_defaults(handler=run_extract)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))
