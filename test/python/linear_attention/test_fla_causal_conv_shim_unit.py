# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contract tests for the opt-in FLA bulk causal-convolution shim."""

import functools
import sys
import types

import pytest
import torch

import cudnn.fla as fla_api
import cudnn.fla.causal_conv1d as conv_shim

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def _reset_telemetry():
    conv_shim.reset_path_counts()
    yield
    conv_shim.reset_path_counts()


@pytest.fixture
def admit_cpu_tensors(monkeypatch):
    monkeypatch.setattr(conv_shim, "_is_cuda_tensor", lambda tensor: True)
    monkeypatch.setattr(conv_shim, "_device_capability", lambda device: (10, 0))


def _operands(*, requires_grad=False):
    x = torch.randn(2, 5, 16, dtype=torch.bfloat16, requires_grad=requires_grad)
    weight = torch.randn(16, 4, dtype=torch.bfloat16, requires_grad=requires_grad)
    return x, weight


@pytest.mark.parametrize(
    "versions,owners,expected",
    [
        ({"flash-linear-attention": "0.5.2", "fla-core": "0.5.2"}, ["flash-linear-attention", "fla-core"], True),
        ({"flash-linear-attention": "0.5.3", "fla-core": "0.5.2"}, ["fla-core"], False),
        ({"flash-linear-attention": "0.5.2", "fla-core": "0.5.3"}, ["fla-core"], False),
        ({"flash-linear-attention": "0.5.2", "fla-core": "0.5.2"}, ["flash-linear-attention"], False),
    ],
)
def test_version_gate_covers_the_meta_and_code_owning_distributions(monkeypatch, versions, owners, expected):
    monkeypatch.setattr(conv_shim, "_installed_distribution_version", versions.get)
    monkeypatch.setattr(conv_shim.metadata, "packages_distributions", lambda: {"fla": owners})

    assert conv_shim.supports_installed_fla() is expected


def test_version_gate_fails_closed_when_python_lacks_package_ownership_api(monkeypatch):
    monkeypatch.setattr(conv_shim, "_installed_distribution_version", lambda name: "0.5.2")
    monkeypatch.delattr(conv_shim.metadata, "packages_distributions", raising=False)

    assert conv_shim.supports_installed_fla() is False


def _call(shim, x, weight, **overrides):
    arguments = {
        "x": x,
        "weight": weight,
        "bias": None,
        "residual": None,
        "initial_state": None,
        "output_final_state": False,
        "activation": "silu",
        "cu_seqlens": None,
        "cu_seqlens_cpu": None,
        "chunk_indices": None,
        "BT": 64,
        "layout_fallback": False,
    }
    arguments.update(overrides)
    return shim(**arguments)


def test_supported_call_uses_native_and_translates_outputs(monkeypatch, admit_cpu_tensors):
    x, weight = _operands()
    expected = (torch.empty_like(x), None)
    calls = []

    def real_fn(**kwargs):
        calls.append(kwargs)
        return "fallback"

    monkeypatch.setattr(conv_shim, "_call_native", lambda *args: expected)
    shim = conv_shim.make_causal_conv1d_fwd(real_fn)

    with torch.no_grad():
        actual = _call(shim, x, weight)

    assert actual is expected
    assert calls == []
    assert conv_shim.last_path() == "native"
    assert conv_shim.path_counts() == {"native": 1}


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"activation": None}, "activation"),
        ({"activation": "relu"}, "activation"),
        ({"bias": torch.zeros(16, dtype=torch.bfloat16)}, "bias-or-residual"),
        ({"residual": torch.zeros(2, 5, 16, dtype=torch.bfloat16)}, "bias-or-residual"),
        ({"output_final_state": None}, "final-state-flag"),
        ({"cu_seqlens": torch.tensor([0, 10], dtype=torch.int32)}, "packed-metadata"),
        ({"cu_seqlens_cpu": torch.tensor([0, 10])}, "packed-metadata"),
        ({"layout_fallback": True}, "layout-fallback"),
        ({"layout_fallback": None}, "layout-fallback"),
    ],
)
def test_unsupported_variants_call_exact_original_once(monkeypatch, admit_cpu_tensors, overrides, reason):
    x, weight = _operands()
    calls = []

    def real_fn(**kwargs):
        calls.append(kwargs)
        return "fallback-result"

    monkeypatch.setattr(conv_shim, "_call_native", lambda *args: pytest.fail("native path should have declined"))
    shim = conv_shim.make_causal_conv1d_fwd(real_fn)
    actual = _call(shim, x, weight, **overrides)

    assert actual == "fallback-result"
    assert len(calls) == 1
    assert calls[0]["x"] is x
    assert calls[0]["weight"] is weight
    for name, value in overrides.items():
        assert calls[0][name] is value
    assert conv_shim.last_path() == f"fallback:{reason}"
    assert conv_shim.path_counts() == {f"fallback:{reason}": 1}


def test_noncontiguous_and_wrong_width_calls_fall_back(monkeypatch, admit_cpu_tensors):
    x, weight = _operands()
    real_calls = []
    shim = conv_shim.make_causal_conv1d_fwd(lambda **kwargs: real_calls.append(kwargs) or "fallback")
    monkeypatch.setattr(conv_shim, "_call_native", lambda *args: pytest.fail("native path should have declined"))

    _call(shim, x[..., ::2], weight[:8])
    assert conv_shim.last_path() == "fallback:shape-or-layout"
    _call(shim, x, weight[:, :3])
    assert conv_shim.last_path() == "fallback:weight-shape"
    assert len(real_calls) == 2


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"chunk_indices": torch.tensor([[0, 0]], dtype=torch.int64)}, "chunk-indices"),
        ({"BT": 17}, "chunk-size"),
        ({"BT": None}, "chunk-size"),
        ({"BT": True}, "chunk-size"),
    ],
)
def test_dense_chunking_variants_fall_back_without_losing_arguments(monkeypatch, admit_cpu_tensors, overrides, reason):
    x, weight = _operands()
    calls = []
    shim = conv_shim.make_causal_conv1d_fwd(lambda **kwargs: calls.append(kwargs) or "fallback")
    monkeypatch.setattr(conv_shim, "_call_native", lambda *args: pytest.fail("native path should have declined"))

    assert _call(shim, x, weight, **overrides) == "fallback"
    assert len(calls) == 1
    for name, value in overrides.items():
        assert calls[0][name] is value
    assert conv_shim.last_path() == f"fallback:{reason}"


@pytest.mark.parametrize(
    "x,initial_state,reason",
    [
        (torch.empty_strided((1, 5, 16), (999, 16, 1), dtype=torch.bfloat16), None, "shape-or-layout"),
        (torch.empty_strided((2, 1, 16), (16, 999, 1), dtype=torch.bfloat16), None, "shape-or-layout"),
        (
            torch.empty((1, 5, 16), dtype=torch.bfloat16),
            torch.empty_strided((1, 16, 4), (999, 4, 1), dtype=torch.bfloat16),
            "initial-state-shape-or-layout",
        ),
    ],
)
def test_singleton_noncanonical_strides_fall_back(monkeypatch, admit_cpu_tensors, x, initial_state, reason):
    # is_contiguous() deliberately ignores singleton-dimension strides. The
    # native descriptor contract does not, so the shim must compare exact strides.
    assert x.is_contiguous()
    if initial_state is not None:
        assert initial_state.is_contiguous()
    weight = torch.randn(x.shape[-1], 4, dtype=torch.bfloat16)
    calls = []
    shim = conv_shim.make_causal_conv1d_fwd(lambda **kwargs: calls.append(kwargs) or "fallback")
    monkeypatch.setattr(conv_shim, "_call_native", lambda *args: pytest.fail("native path should have declined"))

    assert _call(shim, x, weight, initial_state=initial_state) == "fallback"
    assert len(calls) == 1
    assert conv_shim.last_path() == f"fallback:{reason}"


class _TensorSubclass(torch.Tensor):
    pass


def test_tensor_subclass_falls_back_before_raw_pointer_introspection(monkeypatch, admit_cpu_tensors):
    x, weight = _operands()
    x = x.as_subclass(_TensorSubclass)
    calls = []
    shim = conv_shim.make_causal_conv1d_fwd(lambda **kwargs: calls.append(kwargs) or "fallback")
    monkeypatch.setattr(conv_shim, "_call_native", lambda *args: pytest.fail("native path should have declined"))

    assert _call(shim, x, weight) == "fallback"
    assert len(calls) == 1
    assert conv_shim.last_path() == "fallback:tensor-subclass"


@pytest.mark.parametrize(
    "constant,value,shape,with_state,reason",
    [
        ("_MAX_TOTAL_TOKENS", 9, (2, 5, 16), False, "token-index-limit"),
        ("_MAX_SCALAR_GRID_CHANNELS", 15, (1, 5, 16), False, "channel-grid-limit"),
        ("_INT32_MAX", 31, (1, 5, 7), False, "scalar-index-limit"),
        ("_INT32_MAX", 31, (2, 1, 7), True, "state-index-limit"),
    ],
)
def test_native_integer_and_grid_limits_fall_back(monkeypatch, admit_cpu_tensors, constant, value, shape, with_state, reason):
    monkeypatch.setattr(conv_shim, constant, value)
    x = torch.randn(*shape, dtype=torch.bfloat16)
    weight = torch.randn(shape[-1], 4, dtype=torch.bfloat16)
    initial_state = torch.randn(shape[0], shape[-1], 4, dtype=torch.bfloat16) if with_state else None
    calls = []
    shim = conv_shim.make_causal_conv1d_fwd(lambda **kwargs: calls.append(kwargs) or "fallback")
    monkeypatch.setattr(conv_shim, "_call_native", lambda *args: pytest.fail("native path should have declined"))

    assert _call(shim, x, weight, initial_state=initial_state) == "fallback"
    assert len(calls) == 1
    assert conv_shim.last_path() == f"fallback:{reason}"


@pytest.mark.parametrize("mode,reason", [("compile", "compile"), ("autocast", "autocast")])
def test_compiler_and_non_bf16_autocast_decline_before_tensor_introspection(monkeypatch, mode, reason):
    x, weight = _operands()
    real_calls = []
    shim = conv_shim.make_causal_conv1d_fwd(lambda **kwargs: real_calls.append(kwargs) or "fallback")
    monkeypatch.setattr(conv_shim, "_is_compiling", lambda: mode == "compile")
    monkeypatch.setattr(conv_shim, "_cuda_autocast_dtype", lambda: torch.float16 if mode == "autocast" else None)
    monkeypatch.setattr(conv_shim, "_is_cuda_tensor", lambda tensor: pytest.fail("tensor introspection must not run"))

    assert _call(shim, x, weight) == "fallback"
    assert len(real_calls) == 1
    assert conv_shim.last_path() == f"fallback:{reason}"


def test_grad_enabled_low_level_call_falls_back_but_autograd_forward_context_can_run_native(monkeypatch, admit_cpu_tensors):
    x, weight = _operands(requires_grad=True)
    real_calls = []
    shim = conv_shim.make_causal_conv1d_fwd(lambda **kwargs: real_calls.append(kwargs) or "fallback")
    monkeypatch.setattr(conv_shim, "_call_native", lambda *args: (torch.empty_like(x), None))

    assert _call(shim, x, weight) == "fallback"
    assert conv_shim.last_path() == "fallback:grad-enabled-low-level-call"

    with torch.no_grad():
        output, final_state = _call(shim, x, weight)
    assert output.shape == x.shape
    assert final_state is None
    assert conv_shim.last_path() == "native"
    assert len(real_calls) == 1


@pytest.mark.parametrize("error_type", [ImportError, NotImplementedError, conv_shim.cudnn.cudnnGraphNotSupportedError])
def test_typed_native_decline_falls_back(monkeypatch, admit_cpu_tensors, error_type):
    x, weight = _operands()
    real_calls = []

    def decline(*args):
        raise error_type("declined")

    monkeypatch.setattr(conv_shim, "_call_native", decline)
    shim = conv_shim.make_causal_conv1d_fwd(lambda **kwargs: real_calls.append(kwargs) or "fallback")

    with torch.no_grad():
        assert _call(shim, x, weight) == "fallback"
    assert len(real_calls) == 1
    assert conv_shim.last_path() == f"fallback:{error_type.__name__}"


@pytest.mark.parametrize("error_type", [RuntimeError, TypeError, ValueError])
def test_unexpected_native_failure_is_not_hidden(monkeypatch, admit_cpu_tensors, error_type):
    x, weight = _operands()

    def fail(*args):
        raise error_type("broken native launch")

    monkeypatch.setattr(conv_shim, "_call_native", fail)
    shim = conv_shim.make_causal_conv1d_fwd(lambda **kwargs: pytest.fail("unexpected fallback"))

    with torch.no_grad(), pytest.raises(error_type, match="broken native launch"):
        _call(shim, x, weight)
    assert conv_shim.last_path() == f"error:{error_type.__name__}"


def test_path_counts_are_phase_resettable(monkeypatch, admit_cpu_tensors):
    x, weight = _operands()
    monkeypatch.setattr(conv_shim, "_call_native", lambda *args: (torch.empty_like(x), None))
    shim = conv_shim.make_causal_conv1d_fwd(lambda **kwargs: "fallback")

    with torch.no_grad():
        _call(shim, x, weight)
        _call(shim, x, weight, activation=None)
    assert conv_shim.path_counts() == {"native": 1, "fallback:activation": 1}

    conv_shim.reset_path_counts()
    assert conv_shim.path_counts() == {}
    assert conv_shim.last_path() is None


def _fake_stock_fla_conv_module(*, use_runtime_decorators):
    module = types.ModuleType("fla.modules.conv.triton.ops")
    module.__file__ = __file__

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
        return x

    causal_conv1d_fwd.__module__ = module.__name__
    causal_conv1d_fwd.__qualname__ = "causal_conv1d_fwd"
    if use_runtime_decorators:
        backends = pytest.importorskip("fla.ops.backends")
        fla_utils = pytest.importorskip("fla.utils")
        stock = fla_utils.input_guard(no_guard_contiguous=["x"])(causal_conv1d_fwd)
        stock = backends.dispatch("modules")(stock)
    else:

        def decorate(function):
            @functools.wraps(function)
            def wrapper(*args, **kwargs):
                return function(*args, **kwargs)

            return wrapper

        stock = decorate(decorate(causal_conv1d_fwd))
    module.causal_conv1d_fwd = stock
    return module


@pytest.mark.parametrize("dispatch_disabled", [False, True])
def test_stock_callable_gate_accepts_the_runtime_dispatch_topology(monkeypatch, dispatch_disabled):
    backends = pytest.importorskip("fla.ops.backends")

    monkeypatch.setattr(backends, "_DISPATCH_DISABLED", dispatch_disabled)
    module = _fake_stock_fla_conv_module(use_runtime_decorators=True)
    monkeypatch.setattr(fla_api, "_fla_core_owns_module_file", lambda path: True)

    assert fla_api._matches_stock_causal_conv_callable(module, module.causal_conv1d_fwd)


def test_public_conv_activation_rejects_prewrapped_function(monkeypatch):
    module = _fake_stock_fla_conv_module(use_runtime_decorators=False)
    stock = module.causal_conv1d_fwd
    stock_chain = fla_api._function_chain(stock)
    expected_wrapper_codes = tuple(function.__code__ for function in stock_chain[:-1])
    monkeypatch.setattr(fla_api, "_stock_causal_conv_wrapper_codes", lambda: expected_wrapper_codes)
    monkeypatch.setattr(fla_api, "_fla_core_owns_module_file", lambda path: True)
    assert fla_api._matches_stock_causal_conv_callable(module, stock)

    @functools.wraps(stock)
    def third_party(*args, **kwargs):
        return stock(*args, **kwargs)

    module.causal_conv1d_fwd = third_party
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(fla_api, "_conv_supports_installed_fla", lambda: True)
    monkeypatch.setattr(fla_api, "_ORIGINALS", {})

    with pytest.raises(ImportError, match="was replaced or does not match"):
        fla_api.accelerate_fla(verbose=False, targets="conv")

    assert module.causal_conv1d_fwd is third_party
    assert not fla_api.is_accelerated("conv")
