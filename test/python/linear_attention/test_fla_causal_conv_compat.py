# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity and route proof for the opt-in FLA causal-conv forward shim."""

from __future__ import annotations

import importlib
from importlib import metadata

import pytest
import torch

pytest.importorskip("fla.modules.conv.causal_conv1d")

try:
    _FLA_VERSION = metadata.version("flash-linear-attention")
except metadata.PackageNotFoundError:
    _FLA_VERSION = None

from cudnn._causal_conv1d_bulk_arch import FUNCTIONAL_COMPUTE_CAPABILITIES
from cudnn.fla import (
    accelerate_fla,
    conv_last_path,
    conv_path_counts,
    is_accelerated,
    reset_conv_path_counts,
    restore_fla,
)

pytestmark = [
    pytest.mark.L0,
    pytest.mark.gpu_exclusive,
    pytest.mark.xdist_group(name="gpu_exclusive"),
    pytest.mark.skipif(
        not (torch.cuda.is_available() and torch.cuda.get_device_capability() in FUNCTIONAL_COMPUTE_CAPABILITIES),
        reason="the bulk causal-conv forward requires a supported architecture",
    ),
    pytest.mark.skipif(
        _FLA_VERSION != "0.5.2",
        reason="the production causal-conv shim intentionally supports FLA 0.5.2 exactly",
    ),
]


def _run(public_fn, x, weight, upstream, *, initial_state=None, final_upstream=None, backend="triton"):
    output, final_state = public_fn(
        x=x,
        weight=weight,
        initial_state=initial_state,
        output_final_state=final_upstream is not None,
        activation="silu",
        backend=backend,
    )
    loss = (output.float() * upstream).sum()
    inputs = [x, weight]
    if initial_state is not None:
        inputs.append(initial_state)
    if final_upstream is not None:
        assert final_state is not None
        loss = loss + (final_state.float() * final_upstream).sum()
    else:
        assert final_state is None
    gradients = torch.autograd.grad(loss, inputs)
    return output.detach(), None if final_state is None else final_state.detach(), tuple(gradient.detach() for gradient in gradients)


def _clone_inputs(x, weight, initial_state):
    return (
        x.detach().clone().requires_grad_(True),
        weight.detach().clone().requires_grad_(True),
        None if initial_state is None else initial_state.detach().clone().requires_grad_(True),
    )


def _relative_l2(actual, expected):
    return (actual.float() - expected.float()).norm().item() / max(expected.float().norm().item(), 1e-12)


@pytest.mark.parametrize("with_state", [False, True], ids=["stateless", "initial-and-final-state"])
def test_fla_autograd_pairs_native_forward_with_incumbent_backward(with_state):
    public_module = importlib.import_module("fla.modules.conv.causal_conv1d")
    ops_module = importlib.import_module("fla.modules.conv.triton.ops")
    restore_fla(targets="conv")
    original = ops_module.causal_conv1d_fwd

    torch.manual_seed(20260828 + int(with_state))
    batch, tokens, channels = 2, 17, 264
    x = torch.randn(batch, tokens, channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
    initial_state = torch.randn(batch, channels, 4, device="cuda", dtype=torch.bfloat16) if with_state else None
    upstream = torch.randn(batch, tokens, channels, device="cuda", dtype=torch.float32)
    final_upstream = torch.randn(batch, channels, 4, device="cuda", dtype=torch.float32) if with_state else None

    reference_inputs = _clone_inputs(x, weight, initial_state)
    expected = _run(
        public_module.causal_conv1d,
        *reference_inputs[:2],
        upstream,
        initial_state=reference_inputs[2],
        final_upstream=final_upstream,
    )

    assert not is_accelerated("conv")
    try:
        accelerate_fla(verbose=False, targets="conv")
        assert is_accelerated("causal_conv1d_fwd")
        assert ops_module.causal_conv1d_fwd is not original
        reset_conv_path_counts()

        native_inputs = _clone_inputs(x, weight, initial_state)
        output, final_state = public_module.causal_conv1d(
            x=native_inputs[0],
            weight=native_inputs[1],
            initial_state=native_inputs[2],
            output_final_state=with_state,
            activation="silu",
            backend="triton",
        )
        assert conv_last_path() == "native"
        assert conv_path_counts() == {"native": 1}

        loss = (output.float() * upstream).sum()
        if with_state:
            loss = loss + (final_state.float() * final_upstream).sum()
        gradients = tuple(gradient.detach() for gradient in torch.autograd.grad(loss, [value for value in native_inputs if value is not None]))
        actual = output.detach(), None if final_state is None else final_state.detach(), gradients

        # Backward remains FLA-owned. Its activation-free preactivation
        # recompute must be the one and only fallback in this phase.
        assert conv_last_path() == "fallback:activation"
        assert conv_path_counts() == {"native": 1, "fallback:activation": 1}
    finally:
        restore_fla(targets="conv")

    assert ops_module.causal_conv1d_fwd is original
    assert not is_accelerated("conv")
    assert _relative_l2(actual[0], expected[0]) < 2e-2
    if with_state:
        torch.testing.assert_close(actual[1].view(torch.int16), expected[1].view(torch.int16), rtol=0, atol=0)
    else:
        assert actual[1] is expected[1] is None
    assert len(actual[2]) == len(expected[2])
    for actual_gradient, expected_gradient in zip(actual[2], expected[2]):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


def test_packed_metadata_stays_on_the_exact_fla_path():
    public_module = importlib.import_module("fla.modules.conv.causal_conv1d")
    ops_module = importlib.import_module("fla.modules.conv.triton.ops")
    restore_fla(targets="conv")
    original = ops_module.causal_conv1d_fwd

    torch.manual_seed(20260830)
    x = torch.randn(1, 9, 264, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(264, 4, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.tensor((0, 4, 9), device="cuda", dtype=torch.int32)
    try:
        accelerate_fla(verbose=False, targets="conv")
        reset_conv_path_counts()
        output, final_state = public_module.causal_conv1d(
            x=x,
            weight=weight,
            activation="silu",
            backend="triton",
            cu_seqlens=cu_seqlens,
        )
        torch.cuda.synchronize()
        assert output.shape == x.shape
        assert final_state is None
        assert conv_last_path() == "fallback:packed-metadata"
        assert conv_path_counts() == {"fallback:packed-metadata": 1}
    finally:
        restore_fla(targets="conv")
    assert ops_module.causal_conv1d_fwd is original


def test_mix_backend_pairs_native_forward_with_incumbent_cuda_backward():
    cuda_ops = importlib.import_module("fla.modules.conv.cuda.ops")
    if cuda_ops.causal_conv1d_bwd_function is None:
        pytest.skip("the optional causal-conv1d CUDA backward is unavailable")
    public_module = importlib.import_module("fla.modules.conv.causal_conv1d")
    ops_module = importlib.import_module("fla.modules.conv.triton.ops")
    restore_fla(targets="conv")
    original = ops_module.causal_conv1d_fwd

    torch.manual_seed(20260831)
    x = torch.randn(2, 17, 264, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(264, 4, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn_like(x, dtype=torch.float32)
    reference_inputs = _clone_inputs(x, weight, None)
    expected = _run(public_module.causal_conv1d, *reference_inputs[:2], upstream, backend="mix")

    try:
        accelerate_fla(verbose=False, targets="conv")
        reset_conv_path_counts()
        native_inputs = _clone_inputs(x, weight, None)
        actual = _run(public_module.causal_conv1d, *native_inputs[:2], upstream, backend="mix")
        # The mix backend owns its CUDA backward and does not call the Triton
        # activation-recompute forward, so this phase has one native route only.
        assert conv_last_path() == "native"
        assert conv_path_counts() == {"native": 1}
    finally:
        restore_fla(targets="conv")

    assert ops_module.causal_conv1d_fwd is original
    assert _relative_l2(actual[0], expected[0]) < 2e-2
    assert actual[1] is expected[1] is None
    for actual_gradient, expected_gradient in zip(actual[2], expected[2]):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
