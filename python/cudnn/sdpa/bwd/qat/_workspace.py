# SPDX-License-Identifier: Apache-2.0

"""Workspace layout helpers for NVFP4 QAT attention backward."""

from __future__ import annotations

from functools import reduce
from operator import mul

import torch

_WORKSPACE_ALIGNMENT = 16

WorkspaceEntry = tuple[int, tuple[int, ...], torch.dtype]


def _numel(shape: tuple[int, ...]) -> int:
    """Return the number of elements in ``shape``."""
    return reduce(mul, shape, 1)


def _align_up(value: int, alignment: int = _WORKSPACE_ALIGNMENT) -> int:
    """Round ``value`` up to the requested byte alignment."""
    return (value + alignment - 1) // alignment * alignment


def nvfp4_workspace_layout(
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    lse_shape: tuple[int, ...],
) -> tuple[tuple[WorkspaceEntry, ...], int]:
    """Return aligned workspace entries and the total required byte count."""
    entries = []
    offset = 0
    for shape, dtype in (
        (q_shape, torch.bfloat16),
        (k_shape, torch.bfloat16),
        (v_shape, torch.bfloat16),
        (lse_shape, torch.float32),
    ):
        normalized_shape = tuple(shape)
        offset = _align_up(offset)
        entries.append((offset, normalized_shape, dtype))
        offset += _numel(normalized_shape) * dtype.itemsize
    return tuple(entries), _align_up(offset)


FROST_VARIANTS = ("two_kernel", "fused", "fused_nondet")


def frost_variant() -> str:
    """FROST kernel structure: two-kernel recompute (default), or the fused cga1 kernel
    with deterministic / unordered dQ reduce-add.  Experimental switch, read at plan time."""
    import os

    variant = os.environ.get("CUDNN_QAT_FROST_VARIANT", "two_kernel")
    if variant not in FROST_VARIANTS:
        raise ValueError(f"CUDNN_QAT_FROST_VARIANT must be one of {FROST_VARIANTS}, got {variant!r}")
    return variant


def frost_workspace_layout(heads: int, sequence: int, variant: str) -> tuple[tuple[WorkspaceEntry, ...], int]:
    """FROST scratch: fake Q/K/V in BSHD and raw delta in BHS; the fused variants add the
    FP32 dQ accumulator (BHSD) and the per-(b, h, q-tile) ordering semaphores.  O(S) and
    independent of head_chunk.  Shared by api.py (sizing) and _frost.py (carving)."""
    shapes = [
        *((((1, sequence, heads, 128), torch.bfloat16),) * 3),
        ((1, heads, sequence), torch.float32),
    ]
    if variant != "two_kernel":
        shapes += [((1, heads, sequence, 128), torch.float32), ((1, heads, sequence // 128), torch.int32)]
    entries, offset = [], 0
    for shape, dtype in shapes:
        offset = _align_up(offset)
        entries.append((offset, shape, dtype))
        offset += _numel(shape) * dtype.itemsize
    return tuple(entries), _align_up(offset)
