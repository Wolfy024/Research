# Research — Deep Learning Image Watermarking & Encryption

A PyTorch research project exploring two complementary problems in image security:

1. **Invisible Image Watermarking** — hiding a secret image inside a cover image with imperceptible visual change.
2. **Image Encryption** — scrambling image content using chaotic sequences and neural network–based key streams so the image is unrecoverable without the correct keys.

---

## Table of Contents

- [Results](#results)
- [Repository Structure](#repository-structure)
- [Watermark Module](#watermark-module)
- [Encryption Module](#encryption-module)
- [Helper & Pre-processing Utilities](#helper--pre-processing-utilities)
- [Requirements](#requirements)
- [Training](#training)

---

## Results

### Watermarking

| Mode      | SSIM    | PSNR   |
|-----------|---------|--------|
| RGB       | 99.8 %  | 41 dB  |
| Grayscale | 99.99 % | 45 dB  |

> Both metrics are measured between the original cover image and the watermarked output. Higher is better.

---

## Repository Structure

```
Research/
├── Watermark/
│   ├── Model_1/
│   │   ├── Encoder/          # UNET encoder — embeds secret into cover image
│   │   └── BlendModels/      # MAM encoder/decoder — blends bottleneck features
│   └── Model_2/
│       ├── Encoder/          # Improved UNET encoder (includes pre-trained checkpoint)
│       └── GAN.py            # GAN discriminator used during adversarial fine-tuning
│
├── Encryption/
│   ├── Prototype/
│   │   ├── ChaoticSequence/  # Neural model trained to produce chaotic key streams
│   │   ├── Encoder/          # Sequence encoder prototype
│   │   ├── XorSequence/      # XOR key-stream configuration
│   │   └── Compilation/      # End-to-end encrypt / decrypt pipeline (EXR output)
│   ├── Model1/
│   │   └── XOR/              # Linear model trained with an XOR-based encryption loss
│   └── Model2/
│       ├── UNET/             # UNET-based encryption model
│       ├── Encoder/          # Seed-driven key generation utilities
│       └── Compile/          # Bottleneck-level encode / decode pipeline
│
├── HelperFunctions/
│   ├── Calc_Vram.py          # Estimate GPU VRAM required for a given batch/model size
│   └── CheckPath.py          # Validate dataset and output directory paths
│
└── PreProcessingFunctions/
    ├── 3Channelto2Channel.py # Convert RGB images to 2-channel format
    └── RemoveExtraFiles.py   # Strip non-image files (e.g. .mat) from a dataset folder
```

---

## Watermark Module

The watermarking pipeline hides a **secret image** inside a **cover image** such that the output is visually identical to the cover.

### Input Format

| Input        | Colour Space        | Resolution |
|--------------|---------------------|------------|
| Cover image  | RGB                 | 512 × 512  |
| Secret image | RGB **or** Grayscale | 512 × 512  |

### Model_1

- **Encoder** (`Watermark/Model_1/Encoder/`) — a UNET that takes the cover and secret images and produces a watermarked output.
- **BlendModels** (`Watermark/Model_1/BlendModels/`) — a pair of MAM (Multi-Attention Module) encoder and decoder that operate on fused bottleneck features for higher-fidelity blending.

### Model_2

- **Encoder** (`Watermark/Model_2/Encoder/`) — a refined UNET encoder. A pre-trained checkpoint is included under `Encoder/Models/`.
- **GAN** (`Watermark/Model_2/GAN.py`) — a lightweight convolutional discriminator used to push watermarked images closer to the natural image manifold during adversarial training.

---

## Encryption Module

The encryption pipeline transforms an image into an unrecognisable form that can only be decoded with the exact set of keys used during encryption. Encrypted images are stored as lossless **OpenEXR** (`.exr`) files to preserve floating-point precision.

### Prototype

A proof-of-concept pipeline that chains:

1. **Channel diffusion** — each colour channel is independently XOR-diffused with a logistic-map key stream.
2. **Pixel confusion** — pixel positions are permuted using a seed-driven shuffle key.
3. **Reversible decryption** — the same keys applied in reverse order recover the original image exactly.

### Model1 — XOR Linear Model

A feed-forward linear model trained with a custom `EncryptionLoss` to learn an XOR-style transformation from random seeds.

### Model2 — UNET Encryption

A UNET whose bottleneck features are scrambled using tensor-swap operations driven by generated key sequences. The `Compile/` submodule exposes `encode_image` and `Reverse` scripts for the full round-trip.

---

## Helper & Pre-processing Utilities

| File | Purpose |
|------|---------|
| `HelperFunctions/Calc_Vram.py` | Estimate peak GPU memory for a model and batch size before training |
| `HelperFunctions/CheckPath.py` | Assert that required directories exist and are non-empty |
| `PreProcessingFunctions/3Channelto2Channel.py` | Convert RGB images to a 2-channel representation |
| `PreProcessingFunctions/RemoveExtraFiles.py` | Delete non-image files from a dataset directory |

---

## Requirements

- Python ≥ 3.9
- [PyTorch](https://pytorch.org/) with CUDA (recommended)
- torchvision
- OpenEXR + Imath (encryption module only)
- numpy, matplotlib

---

## Training

Each model contains a `Train.py` script and a `config.py` file.  
Default hyperparameters (shared across most models):

| Hyperparameter | Value |
|----------------|-------|
| Optimizer      | Adam  |
| Learning rate  | 2e-4  |
| Batch size     | 16    |
| Epochs         | 20    |
| Train split    | 80 %  |

To train a model, update the dataset path in the relevant `Train.py`, then run:

```bash
cd Watermark/Model_1/Encoder   # or any other model directory
python Train.py
```
