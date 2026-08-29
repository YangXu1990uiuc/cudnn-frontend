# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuDNN bulk-forward adapter for FLA 0.5.2 causal convolution.

The adapter is deliberately opt-in and dense-only. FLA continues to own
backward. Its default Triton backward's activation-free forward recompute falls
back to the exact incumbent implementation; the optional mix backend keeps its
incumbent CUDA backward. Packed calls also fall back: the native primitive
validates device-only offsets with a trap, which is not a safe transparent-shim
contract for metadata that FLA accepted before patching.
"""

from __future__ import annotations

from collections import Counter
import functools
from importlib import metadata

import torch

import cudnn

_SUPPORTED_FLA_VERSION = "0.5.2"
_DECLINE_ERRORS = (ImportError, NotImplementedError, cudnn.cudnnGraphNotSupportedError)
_INT32_MAX = 2**31 - 1
_MAX_TOTAL_TOKENS = _INT32_MAX - 15
_MAX_SCALAR_GRID_CHANNELS = 256 * 65535
_LAST = {"path": None}
_COUNTS = Counter()


def last_path() -> str | None:
    """Return the route selected by the latest shimmed forward call."""

    return _LAST["path"]


def path_counts() -> dict[str, int]:
    """Return a snapshot of route counts since the last explicit reset."""

    return dict(_COUNTS)


def reset_path_counts() -> None:
    """Reset phase-local route telemetry used by tests and model harnesses."""

    _LAST["path"] = None
    _COUNTS.clear()


def _record_path(path: str) -> None:
    _LAST["path"] = path
    _COUNTS[path] += 1


def _installed_distribution_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def supports_installed_fla() -> bool:
    """Whether both the FLA metapackage and code-owning core match."""

    if (
        _installed_distribution_version("flash-linear-attention") != _SUPPORTED_FLA_VERSION
        or _installed_distribution_version("fla-core") != _SUPPORTED_FLA_VERSION
    ):
        return False
    packages_distributions = getattr(metadata, "packages_distributions", None)
    if not callable(packages_distributions):
        return False
    owners = {owner.lower().replace("_", "-") for owner in packages_distributions().get("fla", ())}
    return "fla-core" in owners


def _is_cuda_tensor(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda


def _device_capability(device: torch.device) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device)


def _is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and compiler.is_compiling():
        return True
    dynamo = getattr(torch, "_dynamo", None)
    return bool(dynamo is not None and dynamo.is_compiling())


def _cuda_autocast_dtype():
    try:
        enabled = torch.is_autocast_enabled("cuda")
    except TypeError:
        enabled = torch.is_autocast_enabled()
    if not enabled:
        return None
    try:
        return torch.get_autocast_dtype("cuda")
    except (AttributeError, TypeError):
        return torch.get_autocast_gpu_dtype()


def _tensor_reason(
    tensor,
    *,
    dtype: torch.dtype,
    device: torch.device | None = None,
    shape: tuple[int, ...] | None = None,
    stride: tuple[int, ...] | None = None,
    alignment: int,
) -> str | None:
    if type(tensor) is not torch.Tensor or tensor.layout is not torch.strided:
        return "tensor-subclass"
    if tensor.dtype is not dtype:
        return "dtype"
    if not _is_cuda_tensor(tensor):
        return "non-cuda"
    if device is not None and tensor.device != device:
        return "nonlocal-device"
    if shape is not None and tuple(tensor.shape) != shape:
        return "shape"
    if tensor.numel() == 0 or (stride is not None and tuple(tensor.stride()) != stride):
        return "shape-or-layout"
    if tensor.data_ptr() % alignment:
        return "alignment"
    return None


def _decline_reason(
    x,
    weight,
    bias,
    residual,
    initial_state,
    output_final_state,
    activation,
    cu_seqlens,
    cu_seqlens_cpu,
    chunk_indices,
    BT,
    layout_fallback,
) -> str | None:
    # Avoid tensor/device/data-pointer introspection while a compiler is
    # tracing the incumbent FLA callable.
    if _is_compiling() or torch.jit.is_scripting() or torch.jit.is_tracing():
        return "compile"
    if _cuda_autocast_dtype() not in (None, torch.bfloat16):
        return "autocast"
    if bias is not None or residual is not None:
        return "bias-or-residual"
    if activation not in ("silu", "swish"):
        return "activation"
    if not isinstance(output_final_state, bool):
        return "final-state-flag"
    if cu_seqlens is not None or cu_seqlens_cpu is not None:
        return "packed-metadata"
    if chunk_indices is not None:
        return "chunk-indices"
    if type(BT) is not int or BT != 64:
        return "chunk-size"
    if type(layout_fallback) is not bool or layout_fallback:
        return "layout-fallback"
    if type(x) is not torch.Tensor:
        return "tensor-subclass"
    if x.dim() != 3:
        return "shape"
    batch, tokens, channels = x.shape
    if batch <= 0 or tokens <= 0 or channels <= 0:
        return "shape"
    total_tokens = batch * tokens
    if total_tokens > _MAX_TOTAL_TOKENS:
        return "token-index-limit"
    if channels > _MAX_SCALAR_GRID_CHANNELS:
        return "channel-grid-limit"
    if channels % 8 and total_tokens * channels > _INT32_MAX:
        return "scalar-index-limit"
    if channels % 8 and (initial_state is not None or output_final_state) and batch * channels * 4 > _INT32_MAX:
        return "state-index-limit"
    reason = _tensor_reason(
        x,
        dtype=torch.bfloat16,
        shape=(batch, tokens, channels),
        stride=(tokens * channels, channels, 1),
        alignment=16,
    )
    if reason is not None:
        return reason
    if _device_capability(x.device) != (10, 0):
        return "non-sm100"
    reason = _tensor_reason(
        weight,
        dtype=torch.bfloat16,
        device=x.device,
        shape=(channels, 4),
        stride=(4, 1),
        alignment=16,
    )
    if reason is not None:
        return f"weight-{reason}"
    if initial_state is not None:
        reason = _tensor_reason(
            initial_state,
            dtype=torch.bfloat16,
            device=x.device,
            shape=(batch, channels, 4),
            stride=(channels * 4, 4, 1),
            alignment=16,
        )
        if reason is not None:
            return f"initial-state-{reason}"
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (x, weight, initial_state) if tensor is not None):
        return "grad-enabled-low-level-call"
    return None


def _call_native(x, weight, initial_state, output_final_state):
    from cudnn.causal_conv1d_bulk_sm100 import causal_conv1d_bulk_fwd_wrapper_sm100

    result = causal_conv1d_bulk_fwd_wrapper_sm100(
        x,
        weight,
        initial_state_tensor=initial_state,
        output_final_state=output_final_state,
    )
    final_state = result["final_state_tensor"] if output_final_state else None
    return result["output_tensor"], final_state


def make_causal_conv1d_fwd(real_fn):
    """Wrap FLA's low-level forward with a native fast path and exact fallback."""

    @functools.wraps(real_fn)
    def causal_conv1d_fwd(
        x,
        weight,
        bias,
        residual,
        initial_state=None,
        output_final_state=False,
        activation=None,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        chunk_indices=None,
        BT=64,
        layout_fallback=False,
    ):
        def fallback(reason):
            _record_path(f"fallback:{reason}")
            return real_fn(
                x=x,
                weight=weight,
                bias=bias,
                residual=residual,
                initial_state=initial_state,
                output_final_state=output_final_state,
                activation=activation,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                chunk_indices=chunk_indices,
                BT=BT,
                layout_fallback=layout_fallback,
            )

        reason = _decline_reason(
            x,
            weight,
            bias,
            residual,
            initial_state,
            output_final_state,
            activation,
            cu_seqlens,
            cu_seqlens_cpu,
            chunk_indices,
            BT,
            layout_fallback,
        )
        if reason is not None:
            return fallback(reason)
        try:
            output = _call_native(x, weight, initial_state, output_final_state)
        except _DECLINE_ERRORS as error:
            return fallback(type(error).__name__)
        except (RuntimeError, TypeError, ValueError) as error:
            _record_path(f"error:{type(error).__name__}")
            raise
        _record_path("native")
        return output

    causal_conv1d_fwd.__cudnn_fla_target__ = "causal_conv1d_fwd"
    return causal_conv1d_fwd


__all__ = [
    "last_path",
    "make_causal_conv1d_fwd",
    "path_counts",
    "reset_path_counts",
    "supports_installed_fla",
]
