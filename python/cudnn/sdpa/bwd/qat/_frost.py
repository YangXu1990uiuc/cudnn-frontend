# SPDX-License-Identifier: Apache-2.0

"""Prepared, allocation-free FROST QAT backward orchestration.

Version/capability checks in api.py run before importing this module.
Q/delta and KV preprocessing reuse the public Triton quantizers; two FROST
kernels follow: dK/dV (P fake-quantized for dV, dS consumed in SMEM) and dQ
(dS recomputed, accumulated in TMEM). Nothing S-by-S is materialized and the
result is deterministic.
"""

from dataclasses import dataclass
import math

import cuda.bindings.driver as cuda
import cutlass
import torch

import triton
import triton.language as tl

from ._frost_dq_kernel import compile as compile_dq
from ._frost_fused_kernel import compile as compile_fused
from ._frost_kernel import compile as compile_core
from ._interface import _workspace_tensor
from ._nvfp4 import fake_quantize_kv, fake_quantize_q
from ._workspace import frost_variant, frost_workspace_layout

_CONVERT_BLOCK = 4096


@triton.jit
def _convert_f32_to_bf16(src_ptr, dst_ptr, n_elements, block: tl.constexpr):
    """Final FP32 dQ accumulator -> BF16 dQ (fused variants); allocation-free, precompiled."""
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    valid = offsets < n_elements
    tl.store(dst_ptr + offsets, tl.load(src_ptr + offsets, mask=valid, other=0.0).to(tl.bfloat16), mask=valid)


def workspace_layout(heads: int, sequence: int, head_chunk: int):
    """See _workspace.frost_workspace_layout (shared with api.py sizing)."""
    del head_chunk  # launch granularity only
    return frost_workspace_layout(heads, sequence, frost_variant())


@dataclass(frozen=True)
class PreparedBackward:
    """Own compiled launchers, never change module globals or compile at execute."""

    heads: int
    sequence: int
    head_chunk: int
    entries: tuple
    quant_q: object
    quant_kv: object
    core: object
    core_dq: object
    variant: str
    core_fused: object
    convert: object
    source_strides: tuple
    fake_strides: tuple

    @classmethod
    def compile(cls, heads: int, sequence: int, head_chunk: int):
        entries, _ = workspace_layout(heads, sequence, head_chunk)
        source = (heads * sequence * 128, sequence * 128, 128, 1)
        # Logical BHSD views over the quantizers' BSHD destination.
        fake = (heads * sequence * 128, 128, heads * 128, 1)
        grid = (sequence // 32, heads, 1)
        q_kernel = fake_quantize_q.warmup(
            torch.bfloat16,
            torch.bfloat16,
            *source,
            *fake,
            heads,
            sequence,
            32,
            128,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            grid=grid,
            num_warps=4,
            num_stages=2,
        )
        kv_kernel = fake_quantize_kv.warmup(
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
            *source,
            *fake,
            heads,
            sequence,
            32,
            128,
            grid=grid,
            num_warps=4,
            num_stages=2,
        )
        variant = frost_variant()
        core = core_dq = core_fused = convert = None
        if variant == "two_kernel":
            core = compile_core(1, heads, heads, sequence, sequence, qh_chunk=head_chunk)
            core_dq = compile_dq(1, heads, heads, sequence, sequence, qh_chunk=head_chunk)
        else:
            core_fused = compile_fused(1, heads, heads, sequence, sequence, qh_chunk=head_chunk, deterministic=(variant == "fused"))
            n_dq = heads * sequence * 128
            convert_grid = ((n_dq + _CONVERT_BLOCK - 1) // _CONVERT_BLOCK, 1, 1)
            convert = _convert_f32_to_bf16.warmup(torch.float32, torch.bfloat16, n_dq, _CONVERT_BLOCK, grid=convert_grid, num_warps=4)[convert_grid]
        # Index CompiledKernel now: materializes launch handles without a launch.
        # Execute bypasses Triton's JITFunction/cache-key construction entirely.
        return cls(heads, sequence, head_chunk, entries, q_kernel[grid], kv_kernel[grid], core, core_dq, variant, core_fused, convert, source, fake)

    @torch.no_grad()
    def execute(self, q, k, v, o, do, lse, dq, dk, dv, workspace, scale):
        views = [_workspace_tensor(workspace, entry) for entry in self.entries]
        fake_q, fake_k, fake_v, delta = views[:4]
        stream = torch.cuda.current_stream(q.device).cuda_stream
        self.quant_q(
            q,
            fake_q,
            *self.source_strides,
            *self.fake_strides,
            self.heads,
            self.sequence,
            32,
            128,
            o,
            do,
            delta,
            stream=stream,
        )
        self.quant_kv(
            k,
            v,
            fake_k,
            fake_v,
            *self.source_strides,
            *self.fake_strides,
            self.heads,
            self.sequence,
            32,
            128,
            stream=stream,
        )
        # Only metadata views, including the BSHD views consumed by TMA.
        do_view, dv_view, dk_view, dq_view = (t.permute(0, 2, 1, 3) for t in (do, dv, dk, dq))
        problem = (1, self.heads, self.heads, self.sequence, self.sequence, self.head_chunk)
        scale_log2e = scale * math.log2(math.e)
        if self.variant != "two_kernel":
            # Fused cga1 kernel: dV/dK in TMEM, dQ reduce-added into the FP32 accumulator
            # (kv-tile ordered through the semaphores when deterministic), then converted.
            dq_accum, sem = views[4:6]
            dq_accum.zero_()
            sem.zero_()
            dq_accum_view = dq_accum.permute(0, 2, 1, 3)
            for base in range(0, self.heads, self.head_chunk):
                self.core_fused(
                    fake_q,
                    do_view,
                    fake_k,
                    fake_v,
                    dv_view,
                    dk_view,
                    dq_accum_view,
                    lse,
                    delta,
                    sem,
                    problem,
                    scale,
                    scale_log2e,
                    cutlass.Int32(base),
                    cuda.CUstream(stream),
                )
            self.convert(dq_accum, dq, dq_accum.numel(), _CONVERT_BLOCK, stream=stream)
            return
        for base in range(0, self.heads, self.head_chunk):
            # dK/dV (P fake-quantized for dV; dS recomputed inside the dQ kernel below).
            self.core(
                fake_q,
                do_view,
                fake_k,
                fake_v,
                dv_view,
                dk_view,
                lse,
                delta,
                problem,
                scale,
                scale_log2e,
                1.0,
                1.0,
                1.0,
                scale,
                cutlass.Int32(base),
                cutlass.Int32(self.sequence),
                cuda.CUstream(stream),
            )
            self.core_dq(
                fake_q,
                do_view,
                fake_k,
                fake_v,
                dq_view,
                lse,
                delta,
                problem,
                scale,
                scale_log2e,
                cutlass.Int32(base),
                cuda.CUstream(stream),
            )
