# Model card: SpectralMark showcase

## Summary

SpectralMark embeds a keyed 32-bit provenance identifier into 256×256 RGB
images. It uses fixed orthonormal block-DCT operations, a learned texture-aware
gain network, and learned host-coefficient cancellation. Extraction is blind:
the decoder receives the watermarked image and matching key, not the cover.

## Intended use

- Research on compact, visually subtle provenance identifiers.
- Demonstrations and controlled lossless-PNG workflows.
- A reproducible baseline for keyed, clean-channel bit watermarking.

It is not intended as cryptographic authentication, copyright enforcement,
DRM, covert communication, or a robust watermark for transformed media.

## Training and evaluation

- Training: deterministic 2,000-image subset of COCO val2017.
- Optimization: 500 AdamW steps, batch size 8, seed 24.
- Evaluation: all 100 DIV2K validation images, disjoint from training.
- Input: center-fit 256×256 RGB.
- Output channel: in-memory straight-through 8-bit quantization and lossless PNG.
- Payload: 32 bits.
- Key: wolfy024-provenance-v1 for the committed showcase.

Exact filenames, hashes, configuration, environment, per-image metrics, and
checkpoint hash are recorded in results/showcase_manifest.json and
results/showcase.json.

## Results

| Metric | Value |
|---|---:|
| Bit accuracy | 0.998125 |
| Bit error rate | 0.001875 |
| Exact-message accuracy | 0.97 |
| Cover PSNR | 40.015579 dB |
| Cover global SSIM | 0.998907 |
| Wrong-key bit accuracy | 0.509688 |
| Wrong-key exact-message accuracy | 0.0 |

## Limitations

The model is evaluated only through a clean lossless channel. JPEG, cropping,
resizing, blur, noise, screenshots, print-scan, and adversarial removal are
outside scope. The key is deterministic conditioning, not a secret-key
cryptographic construction. Results are for one checkpoint and one 100-image
cohort, and should not be generalized without broader evaluation.
