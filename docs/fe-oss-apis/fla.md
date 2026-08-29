# FLA Integration Shims

**The FLA integration APIs are experimental and subject to change.**

`cudnn.fla` can replace selected
[`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention)
(FLA) entry points with cuDNN Frontend implementations. The adapters are
process-wide monkeypatches: they cover modules that already exist as well as
modules created after activation. Call them before compiling or tracing a
model, during single-threaded process startup before worker threads begin
importing or executing FLA.

## Activate targets

```python
import cudnn.fla

# Backward-compatible default: Gated Delta Rule and KDA.
cudnn.fla.accelerate_fla()

# Incrementally opt the dense FLA GatedMLP into the fused cuDNN SwiGLU MLP.
cudnn.fla.accelerate_fla(targets="gated_mlp")

# Opt the dense bulk causal-convolution forward into the native CuTeDSL kernel.
cudnn.fla.accelerate_fla(targets="causal_conv1d_fwd")

# A string or iterable is accepted; "gdn", "mlp", and "conv" are aliases.
cudnn.fla.accelerate_fla(targets=("gdn", "gated_mlp", "conv"))
```

`accelerate_fla(verbose=True, *, targets=None)` is incremental and idempotent.
With `targets=None`, it retains the original best-effort behavior and enables
the `gated_delta_rule` and `kda` targets that exist in the installed FLA. An
explicit target selection is atomic: if any requested target cannot be
validated, no new requested target is installed and the raised `ImportError`
includes the rejection reason.

The `gated_mlp` target currently admits exactly FLA 0.5.2's plain, local,
bias-free `swish` `GatedMLP` with fused SwiGLU, contiguous BF16 CUDA inputs and
weights, and an SM100-family device. Unsupported runtime configurations such as
tensor parallelism or DTensor, quantization, LoRA, parametrizations, hooks,
custom linears, other dtypes/layouts/devices, or graph compilation execute the
original FLA method. Typed unsupported-kernel declines also fall back;
unexpected binding, allocation, or launch errors propagate.

The `causal_conv1d_fwd` target currently admits exactly
`flash-linear-attention==0.5.2` with its code-owning `fla-core==0.5.2`
distribution, and FLA's dense,
contiguous BF16 `[batch, tokens, channels]` input with a contiguous BF16
`[channels, 4]` weight, no bias or residual, `silu`/`swish` activation, and a
native functional target (SM80/86/87/89/90/100/103/110/120/121). It supports an optional BF16
`[batch, channels, 4]` initial state and optional final-state output. Packed
sequence metadata, other layouts/dtypes/architectures, non-BF16 autocast,
graph compilation, and direct grad-enabled calls execute the original FLA
function.

The native API selects its schedule from the exact target: SM80-SM90 use the
ordinary-FP32 scalar path, while listed SM100-or-newer targets may use the
packed-FP32 vec8 path for channel widths divisible by eight. Only the B200
schedule has been performance-characterized; other architectures are
functional support and the adapter makes no speedup claim for them.

This target replaces only FLA's low-level forward. During a default Triton
autograd call, cuDNN serves the dense forward while FLA continues to own the
backward. FLA's activation-free preactivation recompute intentionally falls
back to FLA. The optional `mix` backend instead retains its incumbent CUDA
backward and does not recompute through this target. Neither route claims a
cuDNN backward. Packed calls also stay on FLA because the native primitive's
device-side offset validation is not a safe transparent-shim contract for
previously accepted metadata.

## Inspect and restore

```python
import cudnn.fla

cudnn.fla.is_accelerated()             # any live cuDNN FLA target
cudnn.fla.is_accelerated("gated_mlp") # one target ("mlp" also works)
cudnn.fla.is_accelerated("conv")      # the bulk causal-convolution forward

cudnn.fla.mlp_last_path()  # "native", "fallback:<reason>", or "error:<type>"
cudnn.fla.last_path()      # most recent Gated Delta Rule route
cudnn.fla.conv_last_path() # most recent causal-convolution forward route
cudnn.fla.conv_path_counts() # phase-local route counts
cudnn.fla.reset_conv_path_counts()

cudnn.fla.restore_fla(targets="gated_mlp") # restore only the MLP target
cudnn.fla.restore_fla(targets="conv")      # restore only the conv target
cudnn.fla.restore_fla()                    # restore every target owned by cuDNN
```

`restore_fla(*, targets=None)` restores only patches still owned by
`cudnn.fla`; it does not overwrite a later third-party replacement. The route
helpers are diagnostics for tests and benchmarks, not synchronization or
per-thread state.

## Installation

Install FLA separately. The dense MLP adapter is version-gated to the validated
release:

```bash
pip install flash-linear-attention==0.5.2
pip install "nvidia-cudnn-frontend[cutedsl]"
```

The `cutedsl` extra supplies the optional dependencies required by the fused
GEMM path used by the native `gated_mlp` target and the bulk causal-convolution
kernel used by `causal_conv1d_fwd`.
