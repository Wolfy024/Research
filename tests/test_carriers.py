import torch

from neural_watermark.carriers import (
    KeyedBlockLayout,
    block_dct,
    block_idct,
    orthonormal_dct_matrix,
)


def test_block_dct_round_trip() -> None:
    image = torch.rand(2, 1, 32, 32)
    matrix = orthonormal_dct_matrix()
    reconstructed = block_idct(block_dct(image, matrix), matrix, grid_size=4)
    torch.testing.assert_close(reconstructed, image, atol=2e-6, rtol=0)


def test_keyed_layout_is_deterministic_balanced_and_key_specific() -> None:
    first = KeyedBlockLayout(8, 32, "owner-a")
    repeated = KeyedBlockLayout(8, 32, "owner-a")
    different = KeyedBlockLayout(8, 32, "owner-b")

    torch.testing.assert_close(first.bit_indices, repeated.bit_indices)
    torch.testing.assert_close(first.polarities, repeated.polarities)
    assert torch.equal(first.bit_counts, torch.full((8,), 2.0))
    assert not (
        torch.equal(first.bit_indices, different.bit_indices)
        and torch.equal(first.polarities, different.polarities)
        and torch.equal(first.frequency_indices, different.frequency_indices)
    )


def test_insert_select_and_aggregate_are_consistent() -> None:
    layout = KeyedBlockLayout(4, 16, "test")
    values = torch.randn(3, layout.num_blocks)
    coefficients = layout.coefficient_delta(values)
    torch.testing.assert_close(layout.select(coefficients), values)

    messages = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    by_block = layout.message_by_block(messages)
    torch.testing.assert_close(layout.aggregate(by_block), messages)
