"""Validate committed evidence, checkpoint hashes, and README headline values."""

from __future__ import annotations

import json
from pathlib import Path

from neural_watermark.training import sha256_file


def validate_repository(root: Path) -> None:
    result_path = root / "results" / "showcase.json"
    manifest_path = root / "results" / "showcase_manifest.json"
    checkpoint_path = root / "checkpoints" / "showcase.pt"
    readme_path = root / "README.md"
    recovery_figure_path = root / "assets" / "key_recovery_examples.png"

    record = json.loads(result_path.read_text())
    metrics = record["metrics"]["final"]
    per_image = record["metrics"]["per_image"]
    readme = readme_path.read_text()

    assert record["schema_version"] == 1
    assert record["kind"] == "cross_dataset_clean_channel_showcase"
    assert len(per_image) == metrics["images"] == 100
    assert record["checkpoint"]["sha256"] == sha256_file(checkpoint_path)
    assert record["dataset"]["manifest_sha256"] == sha256_file(manifest_path)
    assert recovery_figure_path.is_file()
    assert recovery_figure_path.stat().st_size > 50_000
    assert "assets/key_recovery_examples.png" in readme

    recovery_examples = record["recovery_examples"]
    assert len(recovery_examples) == 3
    assert [row["dataset_index"] for row in recovery_examples] == [0, 50, 99]
    per_image_by_file = {row["file"]: row for row in per_image}
    for row in recovery_examples:
        assert len(row["target_hex"]) == 8
        assert row["target_hex"] == row["matching_key_hex"]
        assert row["target_hex"] != row["wrong_key_hex"]
        assert row["matching_key_bit_accuracy"] == 1.0
        source = per_image_by_file[row["file"]]
        assert row["sha256"] == source["sha256"]
        assert abs(row["psnr"] - source["psnr"]) < 1e-6
        assert abs(row["ssim"] - source["ssim"]) < 1e-6
        assert row["wrong_key_bit_accuracy"] == source["wrong_key_bit_accuracy"]

    expected_text = {
        f"{metrics['bit_accuracy'] * 100:.2f}%",
        f"{round(metrics['exact_message_accuracy'] * metrics['images'])} / "
        f"{metrics['images']}",
        f"{metrics['psnr']:.3f} dB",
        f"{metrics['ssim']:.6f}",
        f"{metrics['wrong_key_bit_accuracy'] * 100:.2f}%",
        f"{record['model']['parameters']:,}",
    }
    missing = sorted(value for value in expected_text if value not in readme)
    if missing:
        raise AssertionError(f"README is missing evidence values: {missing}")

    mean_psnr = sum(row["psnr"] for row in per_image) / len(per_image)
    mean_bit_accuracy = sum(row["bit_accuracy"] for row in per_image) / len(per_image)
    assert abs(mean_psnr - metrics["psnr"]) < 1e-6
    assert abs(mean_bit_accuracy - metrics["bit_accuracy"]) < 1e-9

    scope = record["scope"]
    assert scope["jpeg"] == "not supported or evaluated"
    assert scope["corruption_attacks"] == "not supported or evaluated"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    validate_repository(root)
    print("showcase evidence: valid")


if __name__ == "__main__":
    main()
