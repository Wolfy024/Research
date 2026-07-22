"""Block-DCT transforms and deterministic keyed carrier assignments."""

from __future__ import annotations

import hashlib
import math

import torch
from torch import Tensor, nn

FREQUENCIES: tuple[tuple[int, int], ...] = (
    (1, 3),
    (3, 1),
    (2, 2),
    (2, 3),
    (3, 2),
    (1, 4),
    (4, 1),
    (2, 4),
    (4, 2),
    (3, 3),
)


def key_to_seed(key: str) -> int:
    """Map an arbitrary string key to a stable 63-bit integer seed."""
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def orthonormal_dct_matrix(size: int = 8) -> Tensor:
    positions = torch.arange(size, dtype=torch.float32) + 0.5
    frequencies = torch.arange(size, dtype=torch.float32)[:, None]
    matrix = torch.cos(math.pi * frequencies * positions / size)
    matrix[0] *= math.sqrt(1.0 / size)
    matrix[1:] *= math.sqrt(2.0 / size)
    return matrix


def block_dct(luminance: Tensor, matrix: Tensor, block_size: int = 8) -> Tensor:
    """Convert Bx1xHxW luminance into BxN-blocks of DCT coefficients."""
    if luminance.ndim != 4 or luminance.shape[1] != 1:
        raise ValueError("luminance must have shape [B, 1, H, W]")
    height, width = luminance.shape[-2:]
    if height % block_size or width % block_size:
        raise ValueError("image dimensions must be divisible by block_size")
    blocks = luminance.unfold(2, block_size, block_size).unfold(
        3, block_size, block_size
    )
    blocks = blocks.squeeze(1).reshape(luminance.shape[0], -1, block_size, block_size)
    return torch.einsum("ui,bnij,vj->bnuv", matrix, blocks, matrix)


def block_idct(
    coefficients: Tensor,
    matrix: Tensor,
    grid_size: int,
    block_size: int = 8,
) -> Tensor:
    """Invert BxN block coefficients into Bx1xHxW luminance."""
    if coefficients.ndim != 4 or coefficients.shape[-2:] != (block_size, block_size):
        raise ValueError("coefficients must have shape [B, N, block, block]")
    expected_blocks = grid_size * grid_size
    if coefficients.shape[1] != expected_blocks:
        raise ValueError(f"expected {expected_blocks} blocks")
    blocks = torch.einsum("ui,bnuv,vj->bnij", matrix, coefficients, matrix)
    blocks = blocks.reshape(
        coefficients.shape[0], grid_size, grid_size, block_size, block_size
    )
    image = blocks.permute(0, 1, 3, 2, 4).reshape(
        coefficients.shape[0], 1, grid_size * block_size, grid_size * block_size
    )
    return image


class KeyedBlockLayout(nn.Module):
    """Assign image blocks, DCT frequencies, and polarities from a user key."""

    block_size = 8

    def __init__(self, message_length: int, image_size: int, key: str) -> None:
        super().__init__()
        if image_size % self.block_size:
            raise ValueError("image_size must be divisible by 8")
        self.grid_size = image_size // self.block_size
        self.num_blocks = self.grid_size**2
        if message_length < 1 or message_length > self.num_blocks:
            raise ValueError(
                f"message_length must be in [1, {self.num_blocks}] for this image size"
            )

        generator = torch.Generator().manual_seed(key_to_seed(key))
        block_order = torch.randperm(self.num_blocks, generator=generator)
        balanced_bits = torch.arange(self.num_blocks) % message_length
        bit_indices = torch.empty_like(balanced_bits)
        bit_indices[block_order] = balanced_bits
        frequency_indices = torch.randint(
            len(FREQUENCIES), (self.num_blocks,), generator=generator
        )
        polarities = (
            torch.randint(
                0, 2, (self.num_blocks,), generator=generator, dtype=torch.float32
            )
            .mul(2)
            .sub(1)
        )

        flat_indices = torch.tensor(
            [FREQUENCIES[index] for index in frequency_indices.tolist()],
            dtype=torch.long,
        )
        flat_indices = flat_indices[:, 0] * self.block_size + flat_indices[:, 1]

        self.register_buffer("bit_indices", bit_indices, persistent=False)
        self.register_buffer("frequency_indices", frequency_indices, persistent=False)
        self.register_buffer("flat_frequency_indices", flat_indices, persistent=False)
        self.register_buffer("polarities", polarities, persistent=False)
        self.register_buffer(
            "bit_counts",
            torch.bincount(bit_indices, minlength=message_length).float(),
            persistent=False,
        )
        self.message_length = message_length
        self.image_size = image_size
        self.key = key

    def message_by_block(self, bipolar_bits: Tensor) -> Tensor:
        if bipolar_bits.ndim != 2 or bipolar_bits.shape[1] != self.message_length:
            raise ValueError(
                f"expected messages [B, {self.message_length}], "
                f"got {tuple(bipolar_bits.shape)}"
            )
        return bipolar_bits[:, self.bit_indices]

    def coefficient_delta(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[1] != self.num_blocks:
            raise ValueError(f"expected block values [B, {self.num_blocks}]")
        flat = values.new_zeros(values.shape[0], self.num_blocks, self.block_size**2)
        indices = self.flat_frequency_indices[None, :, None].expand(
            values.shape[0], -1, -1
        )
        flat = flat.scatter(2, indices, values.unsqueeze(-1))
        return flat.reshape(
            values.shape[0], self.num_blocks, self.block_size, self.block_size
        )

    def select(self, coefficients: Tensor) -> Tensor:
        flat = coefficients.flatten(2)
        indices = self.flat_frequency_indices[None, :, None].expand(
            coefficients.shape[0], -1, -1
        )
        return flat.gather(2, indices).squeeze(-1)

    def remove_selected(self, coefficients: Tensor) -> Tensor:
        flat = coefficients.flatten(2)
        indices = self.flat_frequency_indices[None, :, None].expand(
            coefficients.shape[0], -1, -1
        )
        return flat.scatter(2, indices, torch.zeros_like(indices, dtype=flat.dtype))

    def aggregate(self, block_evidence: Tensor) -> Tensor:
        output = block_evidence.new_zeros(block_evidence.shape[0], self.message_length)
        indices = self.bit_indices[None].expand(block_evidence.shape[0], -1)
        output.scatter_add_(1, indices, block_evidence)
        return output / self.bit_counts[None]
