import math

import torch

from neural_watermark.metrics import (
    bit_accuracy,
    bit_error_rate,
    exact_message_accuracy,
    global_ssim,
    psnr,
    theoretical_psnr_for_rms,
)


def test_identical_images_have_perfect_metrics() -> None:
    image = torch.rand(2, 3, 16, 16)
    assert torch.isinf(psnr(image, image)).all()
    torch.testing.assert_close(global_ssim(image, image), torch.ones(2))


def test_psnr_known_offset() -> None:
    reference = torch.zeros(1, 3, 4, 4)
    candidate = torch.full_like(reference, 0.1)
    torch.testing.assert_close(psnr(reference, candidate), torch.tensor([20.0]))
    assert math.isclose(theoretical_psnr_for_rms(0.1), 20.0)


def test_bit_metrics_known_case() -> None:
    logits = torch.tensor([[2.0, -1.0], [-3.0, -2.0]])
    bits = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    assert float(bit_accuracy(logits, bits)) == 0.75
    assert float(bit_error_rate(logits, bits)) == 0.25
    assert float(exact_message_accuracy(logits, bits)) == 0.5
