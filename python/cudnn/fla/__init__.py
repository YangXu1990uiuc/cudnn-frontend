# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuDNN drop-in acceleration for flash-linear-attention (FLA).

``accelerate_fla()`` monkeypatches FLA entry points cuDNN can serve and
transparently calls the original implementation for unsupported configurations.
The backward-compatible no-argument call enables the linear-attention targets
(``gated_delta_rule`` and ``kda``).  Dense MLP and bulk causal-convolution
adapters are intentionally opt-in because they have narrower FLA 0.5.2
contracts::

    import cudnn.fla

    cudnn.fla.accelerate_fla()                    # GDN + KDA, as before
    cudnn.fla.accelerate_fla(targets="gated_mlp") # incremental MLP opt-in
    cudnn.fla.accelerate_fla(targets="causal_conv1d_fwd") # dense bulk forward

Targets can be enabled and restored independently.  Every adapter is
fail-closed: an incompatible installed FLA target rejects explicit activation,
while a runtime configuration outside an activated adapter's validated contract
executes the exact original FLA callable.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from importlib import metadata
import inspect
from pathlib import Path
import sys
from typing import Callable, Iterable

from .causal_conv1d import last_path as conv_last_path
from .causal_conv1d import make_causal_conv1d_fwd
from .causal_conv1d import path_counts as conv_path_counts
from .causal_conv1d import reset_path_counts as reset_conv_path_counts
from .causal_conv1d import supports_installed_fla as _conv_supports_installed_fla
from .gated_delta_rule import make_chunk_gated_delta_rule, last_path
from .gated_mlp import last_path as mlp_last_path
from .gated_mlp import make_gated_mlp_forward
from .gated_mlp import _supports_installed_fla
from .kda import make_chunk_kda

__all__ = [
    "accelerate_fla",
    "conv_last_path",
    "conv_path_counts",
    "is_accelerated",
    "last_path",
    "mlp_last_path",
    "reset_conv_path_counts",
    "restore_fla",
]


@dataclass(frozen=True)
class _PatchSpec:
    module_path: str
    attribute: str
    make_replacement: Callable
    owner_attribute: str | None = None
    default: bool = True


@dataclass(frozen=True)
class _AppliedPatch:
    spec: _PatchSpec
    owner: object
    original: object
    replacement: object


def _function_replacement(factory):
    def make_replacement(module, owner, original):
        del module, owner
        return factory(original)

    return make_replacement


def _gated_mlp_replacement(module, owner, original):
    if not _supports_installed_fla():
        raise ImportError("the cuDNN GatedMLP shim requires flash-linear-attention==0.5.2")
    if owner.__module__ != module.__name__ or owner.__dict__.get("forward") is not original:
        raise ImportError("FLA GatedMLP.forward does not match the expected owning class")
    if original.__module__ != module.__name__ or original.__qualname__ != "GatedMLP.forward" or hasattr(original, "__wrapped__"):
        raise ImportError("FLA GatedMLP.forward was replaced before cuDNN acceleration")
    swiglu_linear_cls = getattr(module, "SwiGLULinear", None)
    if swiglu_linear_cls is None or swiglu_linear_cls.__module__ != module.__name__:
        raise ImportError("FLA GatedMLP does not expose the expected SwiGLULinear helper")
    return make_gated_mlp_forward(original, owner, swiglu_linear_cls)


_CAUSAL_CONV_PARAMETER_CONTRACT = (
    ("x", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
    ("weight", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
    ("bias", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
    ("residual", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
    ("initial_state", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
    ("output_final_state", inspect.Parameter.POSITIONAL_OR_KEYWORD, False),
    ("activation", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
    ("cu_seqlens", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
    ("cu_seqlens_cpu", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
    ("chunk_indices", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
    ("BT", inspect.Parameter.POSITIONAL_OR_KEYWORD, 64),
    ("layout_fallback", inspect.Parameter.POSITIONAL_OR_KEYWORD, False),
)


def _function_chain(function):
    """Return a transparent-wrapper chain, or None for a non-function/cycle."""

    chain = []
    seen = set()
    current = function
    while True:
        if not inspect.isfunction(current) or id(current) in seen:
            return None
        seen.add(id(current))
        chain.append(current)
        if not hasattr(current, "__wrapped__"):
            return tuple(chain)
        current = current.__wrapped__


def _stock_causal_conv_wrapper_codes():
    """Recreate only FLA's decorators to identify their runtime code frames."""

    from fla.ops.backends import dispatch
    from fla.utils import input_guard

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
        del (
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

    wrapped = input_guard(no_guard_contiguous=["x"])(causal_conv1d_fwd)
    wrapped = dispatch("modules")(wrapped)
    chain = _function_chain(wrapped)
    if chain is None:
        raise RuntimeError("could not identify the stock FLA decorator chain")
    return tuple(function.__code__ for function in chain[:-1])


def _fla_core_owns_module_file(module_file: Path) -> bool:
    """Reject PYTHONPATH/edit collisions not owned by one fla-core install."""

    distributions = [
        distribution for distribution in metadata.distributions() if (distribution.metadata.get("Name") or "").lower().replace("_", "-") == "fla-core"
    ]
    if len(distributions) != 1:
        return False
    distribution = distributions[0]
    return any(
        Path(distribution.locate_file(entry)).resolve() == module_file
        for entry in (distribution.files or ())
        if Path(entry).as_posix().endswith("fla/modules/conv/triton/ops.py")
    )


def _matches_stock_causal_conv_callable(module, original) -> bool:
    """Match FLA's raw op plus the decorators active in this exact runtime."""

    try:
        chain = _function_chain(original)
        if chain is None:
            return False
        raw = chain[-1]
        wrapper_codes = tuple(function.__code__ for function in chain[:-1])
        parameters = tuple(
            (parameter.name, parameter.kind, parameter.default) for parameter in inspect.signature(raw, follow_wrapped=False).parameters.values()
        )
        module_file = Path(module.__file__).resolve()
        raw_file = Path(raw.__code__.co_filename).resolve()
        return (
            wrapper_codes == _stock_causal_conv_wrapper_codes()
            and raw.__module__ == module.__name__
            and raw.__qualname__ == "causal_conv1d_fwd"
            and parameters == _CAUSAL_CONV_PARAMETER_CONTRACT
            and raw_file == module_file
            and _fla_core_owns_module_file(module_file)
        )
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return False


def _causal_conv_replacement(module, owner, original):
    del owner
    if not _conv_supports_installed_fla():
        raise ImportError("the cuDNN causal-conv shim requires flash-linear-attention==0.5.2 and fla-core==0.5.2")
    if getattr(module, "causal_conv1d_fwd", None) is not original:
        raise ImportError("FLA causal_conv1d_fwd does not match the expected module owner")
    if not _matches_stock_causal_conv_callable(module, original):
        raise ImportError("FLA causal_conv1d_fwd was replaced or does not match the supported 0.5.2 callable")
    if getattr(original, "__cudnn_fla_target__", None) is not None:
        raise ImportError("FLA causal_conv1d_fwd was replaced before cuDNN acceleration")
    # Importing the native package here makes explicit target activation fail
    # atomically when its optional CuTeDSL dependency is unavailable.
    try:
        from cudnn.causal_conv1d_bulk_sm100 import causal_conv1d_bulk_fwd_wrapper_sm100  # noqa: F401
        from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old
    except (ImportError, OSError) as error:
        raise ImportError(f"the cuDNN bulk causal-conv forward is unavailable: {error}") from error
    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        raise ImportError("the cuDNN bulk causal-conv forward requires nvidia-cutlass-dsl>=4.7.0")
    return make_causal_conv1d_fwd(original)


_TARGETS = {
    "gated_delta_rule": _PatchSpec(
        "fla.ops.gated_delta_rule",
        "chunk_gated_delta_rule",
        _function_replacement(make_chunk_gated_delta_rule),
    ),
    "kda": _PatchSpec(
        "fla.ops.kda",
        "chunk_kda",
        _function_replacement(make_chunk_kda),
    ),
    "gated_mlp": _PatchSpec(
        "fla.modules.mlp",
        "forward",
        _gated_mlp_replacement,
        owner_attribute="GatedMLP",
        default=False,
    ),
    "causal_conv1d_fwd": _PatchSpec(
        "fla.modules.conv.triton.ops",
        "causal_conv1d_fwd",
        _causal_conv_replacement,
        default=False,
    ),
}
_ALIASES = {"gdn": "gated_delta_rule", "mlp": "gated_mlp", "conv": "causal_conv1d_fwd"}
_DEFAULT_TARGETS = tuple(name for name, spec in _TARGETS.items() if spec.default)
_ORIGINALS: dict[str, _AppliedPatch] = {}


def _canonical_target(target: str) -> str:
    if not isinstance(target, str):
        raise TypeError(f"FLA acceleration target must be a string, got {type(target).__name__}")
    target = _ALIASES.get(target, target)
    if target not in _TARGETS:
        choices = ", ".join(_TARGETS)
        raise ValueError(f"unknown FLA acceleration target {target!r}; expected one of: {choices}")
    return target


def _normalize_targets(targets: str | Iterable[str] | None, *, default: Iterable[str]) -> tuple[str, ...]:
    if targets is None:
        requested = tuple(default)
    elif isinstance(targets, str):
        requested = (targets,)
    else:
        requested = tuple(targets)
    if not requested:
        raise ValueError("at least one FLA acceleration target is required")
    canonical = {_canonical_target(target) for target in requested}
    # Registry order makes logging/restoration deterministic even if a set was
    # supplied by the caller.
    return tuple(target for target in _TARGETS if target in canonical)


def is_accelerated(target: str | None = None) -> bool:
    """Whether any target, or one named target, is currently patched."""
    if target is None:
        return any(getattr(applied.owner, applied.spec.attribute, None) is applied.replacement for applied in _ORIGINALS.values())
    applied = _ORIGINALS.get(_canonical_target(target))
    return applied is not None and getattr(applied.owner, applied.spec.attribute, None) is applied.replacement


def _drop_displaced_patch(target: str) -> None:
    """Forget a patch whose owner was replaced without clobbering the new owner."""
    applied = _ORIGINALS.pop(target, None)
    if applied is None:
        return
    if applied.spec.owner_attribute is None:
        _rebind_everywhere(applied.spec.attribute, applied.replacement, applied.original)


def _rebind_everywhere(fn_name: str, original, replacement) -> None:
    """Rebind imports that captured a patched module-level function by reference."""
    for module in list(sys.modules.values()):
        if module is None:
            continue
        try:
            if getattr(module, fn_name, None) is original:
                setattr(module, fn_name, replacement)
        except Exception:
            # Some modules raise on getattr of arbitrary names; skip them.
            continue


def accelerate_fla(verbose: bool = True, *, targets: str | Iterable[str] | None = None) -> None:
    """Patch selected FLA targets, incrementally and idempotently.

    ``targets=None`` preserves the original behavior and enables only GDN/KDA.
    Use ``targets="gated_mlp"`` (or ``"mlp"``) for the opt-in dense MLP
    adapter and ``targets="causal_conv1d_fwd"`` (or ``"conv"``) for the
    dense bulk causal-convolution forward. A string or iterable is accepted.
    """
    requested = _normalize_targets(targets, default=_DEFAULT_TARGETS)
    available = {target for target in requested if is_accelerated(target)}
    resolved = []
    missing = []
    rejection_reasons = {}

    for target in requested:
        if is_accelerated(target):
            continue
        _drop_displaced_patch(target)
        spec = _TARGETS[target]
        try:
            module = importlib.import_module(spec.module_path)
        except ImportError as error:
            missing.append(target)
            rejection_reasons[target] = str(error) or f"cannot import {spec.module_path}"
            continue
        owner = getattr(module, spec.owner_attribute, None) if spec.owner_attribute is not None else module
        if owner is None:
            missing.append(target)
            rejection_reasons[target] = f"{spec.module_path} has no {spec.owner_attribute} owner"
            continue
        original = getattr(owner, spec.attribute, None)
        if original is None:
            missing.append(target)
            rejection_reasons[target] = f"the target owner has no {spec.attribute} attribute"
            continue
        try:
            replacement = spec.make_replacement(module, owner, original)
        except ImportError as error:
            missing.append(target)
            rejection_reasons[target] = str(error) or "the installed target does not match the supported contract"
            continue
        resolved.append((target, spec, owner, original, replacement))

    # An explicit selection is a contract: never silently apply only a subset.
    # The legacy no-target call retains its best-effort GDN/KDA behavior.
    if targets is not None and missing:
        details = ", ".join(f"{target} ({rejection_reasons[target]})" if target in rejection_reasons else target for target in missing)
        raise ImportError(f"accelerate_fla() could not enable FLA target(s): {details}")

    newly_patched = []
    for target, spec, owner, original, replacement in resolved:
        if spec.owner_attribute is None:
            _rebind_everywhere(spec.attribute, original, replacement)
        else:
            setattr(owner, spec.attribute, replacement)
        _ORIGINALS[target] = _AppliedPatch(spec, owner, original, replacement)
        newly_patched.append(target)
        available.add(target)

    if not available and not newly_patched:
        names = ", ".join(requested)
        raise ImportError(f"accelerate_fla() could not find supported FLA target(s): {names}")
    if verbose and newly_patched:
        print(f"[cudnn.fla] accelerated FLA {', '.join(newly_patched)} with cuDNN (SM100); " "unsupported configs fall back to FLA.")


def restore_fla(*, targets: str | Iterable[str] | None = None) -> None:
    """Undo all active patches, or only the selected targets."""
    if targets is None:
        requested = tuple(target for target in _TARGETS if target in _ORIGINALS)
    else:
        requested = _normalize_targets(targets, default=())
    for target in requested:
        applied = _ORIGINALS.pop(target, None)
        if applied is None:
            continue
        spec, owner, original, replacement = (
            applied.spec,
            applied.owner,
            applied.original,
            applied.replacement,
        )
        if spec.owner_attribute is None:
            _rebind_everywhere(spec.attribute, replacement, original)
        if getattr(owner, spec.attribute, None) is replacement:
            setattr(owner, spec.attribute, original)
