# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""The prepared THD f16 launch (``cudnn.sdpa.fwd.prepared``) behind ``graph.execute()``.

One THD f16 plan, executed through the graph's normalized VariantPack: the plan is prepared,
capacities come from the caller's observed spans (not the graph's ragged declaration), every
call binds an independent frame, outputs are fully written, degenerate inputs are handled or
rejected before any launch, and execute allocates nothing and never synchronizes.
"""

from __future__ import annotations

import math

import pytest
import torch

import cudnn
from cudnn.sdpa.fwd import prepared as prep_mod
from cudnn.sdpa.fwd.engines import engine_name
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0]

DEV = torch.device("cuda")


def _thd_graph(b, ql, kl, hq, hk, d, *, ragged_batch_stride=None, causal=True):
    """A THD bf16 graph the way FlashInfer declares it: BHSD dims with ragged offsets, cu_seq_len
    lengths, token-major Stats. ``ragged_batch_stride`` mimics FlashInfer's small declared batch
    stride (the declaration's span is then far below the buffer's)."""
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q_bs = ragged_batch_stride if ragged_batch_stride is not None else ql * hq * d
    kv_bs = ragged_batch_stride if ragged_batch_stride is not None else kl * hk * d
    tq = g.tensor(dim=[b, hq, ql, d], stride=[q_bs, d, hq * d, 1], data_type=cudnn.data_type.BFLOAT16, name="q")
    tk = g.tensor(dim=[b, hk, kl, d], stride=[kv_bs, d, hk * d, 1], data_type=cudnn.data_type.BFLOAT16, name="k")
    tv = g.tensor(dim=[b, hk, kl, d], stride=[kv_bs, d, hk * d, 1], data_type=cudnn.data_type.BFLOAT16, name="v")
    i32 = cudnn.data_type.INT32
    t_cu_q = g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=i32, name="cu_q")
    t_cu_kv = g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=i32, name="cu_kv")
    off_q = g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=i32, name="off_q")
    off_kv = g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=i32, name="off_kv")
    off_lse = g.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=i32, name="off_lse")
    tq.set_ragged_offset(off_q)
    tk.set_ragged_offset(off_kv)
    tv.set_ragged_offset(off_kv)
    to, ts = g.sdpa(
        name="sdpa",
        q=tq,
        k=tk,
        v=tv,
        generate_stats=True,
        attn_scale=1.0 / math.sqrt(d),
        use_causal_mask=causal,
        use_padding_mask=True,
        cu_seq_len_q=t_cu_q,
        cu_seq_len_kv=t_cu_kv,
        max_total_seq_len_q=b * ql,
        max_total_seq_len_kv=b * kl,
    )
    to.set_output(True).set_dim([b, hq, ql, d]).set_stride([q_bs, d, hq * d, 1]).set_ragged_offset(off_q)
    ts.set_output(True).set_dim([b, hq, ql, 1]).set_stride([ql * hq, 1, hq, 1]).set_data_type(cudnn.data_type.FLOAT).set_ragged_offset(off_lse)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    want = engine_name()
    g.select_plan(next(i for i, n in enumerate(names) if n == want or n.startswith(want + "[")))
    g.check_support()
    g.build_plans()
    return g, dict(q=tq, k=tk, v=tv, o=to, stats=ts, cu_q=t_cu_q, cu_kv=t_cu_kv, off_q=off_q, off_kv=off_kv, off_lse=off_lse)


def _buffers(b, ql, kl, hq, hk, d, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(b * ql, hq, d, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(b * kl, hk, d, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(b * kl, hk, d, device=DEV, dtype=torch.bfloat16)
    o = torch.empty(b * ql, hq, d, device=DEV, dtype=torch.bfloat16)
    lse = torch.empty(b * ql, hq, device=DEV, dtype=torch.float32)
    cu_q = (torch.arange(0, b + 1, device=DEV, dtype=torch.int32) * ql).contiguous()
    cu_kv = (torch.arange(0, b + 1, device=DEV, dtype=torch.int32) * kl).contiguous()
    return dict(
        q=q,
        k=k,
        v=v,
        o=o,
        lse=lse,
        cu_q=cu_q,
        cu_kv=cu_kv,
        off_q=(cu_q * hq * d).to(torch.int32),
        off_kv=(cu_kv * hk * d).to(torch.int32),
        off_lse=(cu_q * hq).to(torch.int32),
    )


def _pack(t, bufs):
    return {
        t["q"]: bufs["q"],
        t["k"]: bufs["k"],
        t["v"]: bufs["v"],
        t["o"]: bufs["o"],
        t["stats"]: bufs["lse"],
        t["cu_q"]: bufs["cu_q"],
        t["cu_kv"]: bufs["cu_kv"],
        t["off_q"]: bufs["off_q"],
        t["off_kv"]: bufs["off_kv"],
        t["off_lse"]: bufs["off_lse"],
    }


def _reference(bufs, b, ql, kl, hq, hk, d, causal=True):
    """fp32 causal (or full) attention per sequence with GQA broadcast; returns O (T, H, D) and LSE (T, H)."""
    q, k, v = bufs["q"].float(), bufs["k"].float(), bufs["v"].float()
    o = torch.empty(b * ql, hq, d, device=DEV)
    lse = torch.empty(b * ql, hq, device=DEV)
    g = hq // hk
    for i in range(b):
        qi = q[i * ql : (i + 1) * ql].transpose(0, 1)  # (H, ql, d)
        ki = k[i * kl : (i + 1) * kl].transpose(0, 1).repeat_interleave(g, 0)
        vi = v[i * kl : (i + 1) * kl].transpose(0, 1).repeat_interleave(g, 0)
        s = qi @ ki.transpose(1, 2) / math.sqrt(d)
        if causal:  # use_causal_mask: top-left aligned diagonal
            row = torch.arange(ql, device=DEV).view(-1, 1)
            col = torch.arange(kl, device=DEV).view(1, -1)
            s = s.masked_fill(col > row, float("-inf"))
        lse[i * ql : (i + 1) * ql] = torch.logsumexp(s, dim=-1).transpose(0, 1)
        o[i * ql : (i + 1) * ql] = (torch.softmax(s, dim=-1) @ vi).transpose(0, 1)
    return o, lse


def _plan(g):
    return g._compiled_plans[g._plan_index]


class _Recorder:
    """Records every frame the prepared launch hands to the positional entry."""

    def __init__(self, spec):
        self.spec, self.frames, self._fn = spec, [], spec.fn
        spec.fn = self

    def __call__(self, *frame):
        self.frames.append(dict(zip(self.spec.order, frame)))
        return self._fn(*frame)

    def restore(self):
        self.spec.fn = self._fn


@requires_pre_rubin_blackwell
@requires_dsl
def test_thd_f16_plan_is_prepared_and_binds_the_variant_pack():
    b, ql, kl, hq, hk, d = 4, 4, 64, 8, 2, 128
    g, t = _thd_graph(b, ql, kl, hq, hk, d)
    plan = _plan(g)
    assert isinstance(plan._prepared, prep_mod.PreparedThdLaunch)
    assert plan.takes_variant_pack is True
    bufs = _buffers(b, ql, kl, hq, hk, d)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    g.execute(_pack(t, bufs), ws)
    torch.cuda.synchronize()
    o_ref, lse_ref = _reference(bufs, b, ql, kl, hq, hk, d)
    torch.testing.assert_close(bufs["o"].float(), o_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(bufs["lse"], lse_ref, atol=1e-3, rtol=1e-3)


@requires_pre_rubin_blackwell
@requires_dsl
def test_capacity_comes_from_the_observed_span_not_the_ragged_declaration():
    """FlashInfer declares Q with a batch stride far below one batch's rows; the pack re-describes the
    caller's (T, H, D) buffer with that geometry (graph_described). The token capacity must still be
    the buffer's T, or the TMA extent truncates and live rows go unwritten (the first version of
    this path bound 67 instead of 256)."""
    b, ql, kl, hq, hk, d = 64, 4, 64, 8, 2, 128
    g, t = _thd_graph(b, ql, kl, hq, hk, d, ragged_batch_stride=1024)
    plan = _plan(g)
    rec = _Recorder(plan._prepared.spec)
    try:
        bufs = _buffers(b, ql, kl, hq, hk, d)
        ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
        bufs["o"].fill_(float("nan"))
        bufs["lse"].fill_(float("nan"))
        g.execute(_pack(t, bufs), ws)
        torch.cuda.synchronize()
    finally:
        rec.restore()
    assert len(rec.frames) == 1
    problem = rec.frames[0]["problem_size"]
    assert problem[3] == b * ql and problem[4] == b * kl, problem
    assert not torch.isnan(bufs["o"]).any() and not torch.isnan(bufs["lse"]).any(), "poisoned outputs must be fully overwritten"
    o_ref, lse_ref = _reference(bufs, b, ql, kl, hq, hk, d)
    torch.testing.assert_close(bufs["o"].float(), o_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(bufs["lse"], lse_ref, atol=1e-3, rtol=1e-3)


@requires_pre_rubin_blackwell
@requires_dsl
def test_each_call_binds_its_own_buffers():
    b, ql, kl, hq, hk, d = 4, 4, 64, 8, 2, 128
    g, t = _thd_graph(b, ql, kl, hq, hk, d)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    rec = _Recorder(_plan(g)._prepared.spec)
    try:
        a, c = _buffers(b, ql, kl, hq, hk, d, seed=1), _buffers(b, ql, kl, hq, hk, d, seed=2)
        g.execute(_pack(t, a), ws)
        g.execute(_pack(t, c), ws)
        torch.cuda.synchronize()
    finally:
        rec.restore()
    fa, fc = rec.frames
    assert fa["q_ptr"] == a["q"].data_ptr() and fc["q_ptr"] == c["q"].data_ptr() and fa["q_ptr"] != fc["q_ptr"]
    assert fa["o_ptr"] != fc["o_ptr"] and fa["lse_ptr"] != fc["lse_ptr"]
    for bufs in (a, c):
        o_ref, lse_ref = _reference(bufs, b, ql, kl, hq, hk, d)
        torch.testing.assert_close(bufs["o"].float(), o_ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(bufs["lse"], lse_ref, atol=1e-3, rtol=1e-3)


@requires_pre_rubin_blackwell
@requires_dsl
def test_shrinking_lengths_and_zero_capacity():
    """Shorter ragged lengths on the same buffers leave rows past each length untouched (the plan
    reads lengths on the device); an empty Q buffer binds no launch at all."""
    b, ql, kl, hq, hk, d = 4, 8, 64, 8, 2, 128
    g, t = _thd_graph(b, ql, kl, hq, hk, d)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    bufs = _buffers(b, ql, kl, hq, hk, d)
    live = 3  # tokens per sequence actually present: the packed buffer holds b*live live rows, the rest is slack
    cu_q = (torch.arange(0, b + 1, device=DEV, dtype=torch.int32) * live).contiguous()
    bufs["cu_q"], bufs["off_q"], bufs["off_lse"] = cu_q, (cu_q * hq * d).to(torch.int32), (cu_q * hq).to(torch.int32)
    bufs["o"].fill_(7.0)
    bufs["lse"].fill_(7.0)
    g.execute(_pack(t, bufs), ws)
    torch.cuda.synchronize()
    o = bufs["o"].float()
    assert not (o[: b * live] == 7.0).all(dim=-1).any(), "live rows must be written"
    assert (o[b * live :] == 7.0).all(), "rows past the packed total are not the kernel's to write"
    # zero capacity: no Q token addressable -> bind returns None, nothing is launched
    rec = _Recorder(_plan(g)._prepared.spec)
    try:
        empty = dict(
            bufs,
            q=torch.empty(0, hq, d, device=DEV, dtype=torch.bfloat16),
            o=torch.empty(0, hq, d, device=DEV, dtype=torch.bfloat16),
            lse=torch.empty(0, hq, device=DEV),
        )
        g.execute(_pack(t, empty), ws)
        torch.cuda.synchronize()
    finally:
        rec.restore()
    assert rec.frames == []


@requires_pre_rubin_blackwell
@requires_dsl
def test_bare_address_and_wrong_device_are_rejected_before_launch():
    b, ql, kl, hq, hk, d = 4, 4, 64, 8, 2, 128
    g, t = _thd_graph(b, ql, kl, hq, hk, d)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    bufs = _buffers(b, ql, kl, hq, hk, d)
    rec = _Recorder(_plan(g)._prepared.spec)
    try:
        with pytest.raises(ValueError, match="bare address"):
            g.execute(_pack(t, dict(bufs, q=bufs["q"].data_ptr())), ws)
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            g.execute(_pack(t, dict(bufs, k=bufs["k"].cpu())), ws)
        with pytest.raises((ValueError, RuntimeError, TypeError)):  # auxiliary roles carry the same device rule
            g.execute(_pack(t, dict(bufs, cu_q=bufs["cu_q"].cpu())), ws)
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            g.execute(_pack(t, dict(bufs, lse=bufs["lse"].cpu())), ws)
    finally:
        rec.restore()
    assert rec.frames == [], "a rejected call must not reach the launch"


@requires_pre_rubin_blackwell
@requires_dsl
def test_zero_kv_clamp_and_stride_mismatch():
    """All-zero KV lengths bind the V stub the spec allocated at build (no allocation, no first-use
    initialization during execute); a THD operand whose effective strides differ from the plan's
    declared ones is rejected before launch (stride override is not bound yet)."""
    b, ql, kl, hq, hk, d = 4, 4, 64, 8, 2, 128
    g, t = _thd_graph(b, ql, kl, hq, hk, d)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    bufs = _buffers(b, ql, kl, hq, hk, d)
    spec = _plan(g)._prepared.spec
    assert "v_stub" in spec._dummies and "sinks" in spec._dummies, "resources exist before the first execute"
    stub_before = spec.dummy("v_stub")
    rec = _Recorder(spec)
    try:
        zero_kv = dict(bufs, k=torch.empty(0, hk, d, device=DEV, dtype=torch.bfloat16), v=torch.empty(0, hk, d, device=DEV, dtype=torch.bfloat16))
        zero_kv["cu_kv"] = torch.zeros(b + 1, device=DEV, dtype=torch.int32)
        zero_kv["off_kv"] = torch.zeros(b + 1, device=DEV, dtype=torch.int32)
        g.execute(_pack(t, zero_kv), ws)
        torch.cuda.synchronize()
    finally:
        rec.restore()
    frame = rec.frames[-1]
    assert frame["problem_size"][4] == 1 and frame["v_ptr"] == stub_before and frame["k_ptr"] == frame["q_ptr"]
    assert spec.dummy("v_stub") == stub_before
    # a K with a different head stride (a (T, 2H, D) slab sliced to H heads) is a stride override: declined
    wide_k = torch.randn(b * kl, 2 * hk, d, device=DEV, dtype=torch.bfloat16)[:, :hk]
    rec = _Recorder(spec)
    try:
        with pytest.raises(ValueError, match="stride"):
            g.execute(_pack(t, dict(bufs, k=wide_k)), ws)
    finally:
        rec.restore()
    assert rec.frames == []


@requires_pre_rubin_blackwell
@requires_dsl
def test_execute_allocates_nothing_and_never_synchronizes():
    """Warmed-path check: after one execute, further executes add no torch allocation and trigger no
    torch-visible synchronization. Driver-side allocation is excluded by construction (the spec
    allocates its resources at build; see test_zero_kv_clamp_and_stride_mismatch)."""
    b, ql, kl, hq, hk, d = 4, 4, 64, 8, 2, 128
    g, t = _thd_graph(b, ql, kl, hq, hk, d)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    bufs = _buffers(b, ql, kl, hq, hk, d)
    pack = _pack(t, bufs)
    g.execute(pack, ws)  # warm: dummies, indices
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(5):
            g.execute(pack, ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before


@requires_pre_rubin_blackwell
@requires_dsl
def test_graph_and_standalone_execute_bind_the_same_frame():
    """The graph plan and the adapter's execute() are the same core: identical frames, up to the stream."""
    b, ql, kl, hq, hk, d = 4, 4, 64, 8, 2, 128
    g, t = _thd_graph(b, ql, kl, hq, hk, d)
    ws = torch.empty(max(g.get_workspace_size(), 1), device=DEV, dtype=torch.uint8)
    bufs = _buffers(b, ql, kl, hq, hk, d)
    plan = _plan(g)
    rec = _Recorder(plan._prepared.spec)
    try:
        g.execute(_pack(t, bufs), ws)
        torch.cuda.synchronize()
        prepared_frame = rec.frames[-1]
        plan._prepared, plan.takes_variant_pack = None, False
        try:
            g.execute(_pack(t, bufs), ws)
            torch.cuda.synchronize()
        finally:
            plan._prepared, plan.takes_variant_pack = rec.spec and plan._compiled.prepared, True
        standalone_frame = rec.frames[-1]
    finally:
        rec.restore()
    assert len(rec.frames) == 2
    for name in rec.spec.order:
        if name == "stream":
            continue
        assert str(prepared_frame[name]) == str(standalone_frame[name]), name
