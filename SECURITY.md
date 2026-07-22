# Security and threat model

SpectralMark is a research watermark, not a security boundary.

## What the key does

The user key deterministically chooses the payload-bit assignment, DCT
frequency, and polarity for every 8×8 image block. A mismatched key produces
near-chance recovery in the committed evaluation.

## What the key does not do

- It does not encrypt the payload or image.
- It does not provide a digital signature or proof of identity.
- It does not prevent key guessing, watermark removal, or forgery.
- It does not authenticate metadata or bind an image to an external record.

For authenticity, sign the image or a canonical image digest with a reviewed
cryptographic signature scheme. Treat the watermark only as an auxiliary
signal.

## Supported channel

Only 8-bit lossless PNG storage is evaluated. JPEG, geometric transforms,
filtering, screenshots, print-scan, and intentional attacks are unsupported.
Do not rely on recovery after any of these operations.

## Reporting issues

Open a private GitHub security advisory for a vulnerability in the software.
Model-quality limitations and research questions can use the public issue
tracker.
