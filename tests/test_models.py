import torch

from neural_watermark.losses import WatermarkObjective
from neural_watermark.models import ModelConfig, NeuralWatermarker


def small_model(key: str = "test-key") -> NeuralWatermarker:
    return NeuralWatermarker(
        ModelConfig(
            image_size=16,
            message_length=4,
            key=key,
            hidden_channels=4,
            max_coefficient_delta=0.12,
            quantize=True,
        )
    )


def test_forward_is_blind_quantized_and_differentiable() -> None:
    model = small_model()
    cover = torch.rand(2, 3, 16, 16)
    bits = torch.randint(0, 2, (2, 4)).float()
    outputs = model(cover, bits)

    assert outputs["watermarked"].shape == cover.shape
    assert outputs["logits"].shape == bits.shape
    assert model.reveal(outputs["watermarked"]).shape == bits.shape
    assert torch.all(
        (outputs["watermarked"] * 255).round() == outputs["watermarked"] * 255
    )

    objective = WatermarkObjective(model.layout.grid_size)
    loss, _ = objective(cover, bits, outputs)
    loss.backward()
    assert model.gain_network.features[0].weight.grad is not None
    assert model.host_predictor.network[0].weight.grad is not None


def test_wrong_key_changes_blind_decode() -> None:
    first = small_model("first")
    wrong = small_model("wrong")
    wrong.load_state_dict(first.state_dict())
    cover = torch.rand(2, 3, 16, 16)
    bits = torch.randint(0, 2, (2, 4)).float()

    marked = first.embed(cover, bits)["watermarked"]
    first_logits = first.reveal_details(marked)["logits"]
    wrong_logits = wrong.reveal_details(marked)["logits"]
    assert not torch.equal(first_logits, wrong_logits)


def test_model_rejects_bad_shapes_and_bits() -> None:
    model = small_model()
    cover = torch.rand(1, 3, 16, 16)
    try:
        model(cover, torch.ones(1, 5))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong message size must fail")

    try:
        model(cover, torch.full((1, 4), 2.0))
    except ValueError:
        pass
    else:
        raise AssertionError("non-binary-range message must fail")
