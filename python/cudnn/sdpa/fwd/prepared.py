# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Prepared launch for the SM100 THD (packed / ragged) f16 forward plan.

Everything a launch needs that the plan already fixed is resolved once, here:
the positional argument template of the explicit host entry (plan constants
filled in), which argument slots each variant-pack operand feeds, the
validation rule per operand, the workspace layout. ``execute`` is then
lookups, integer writes into a per-call copy of the template, the declared
per-call operations (the padded-Stats ``-inf`` seed) and one positional
tvm-ffi call: no ``cute.Pointer`` / ``cute.Tensor`` objects, no keyword
wrapper, no device allocation, no compile.
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Dict, List, Optional

from cudnn.frost import buffers as _buffers
from cudnn.frost.compiled_cache import positional_entry

_ALIGN_TMA = 16
_ALIGN_F32 = 4


def _dtype_name(op) -> str:
    return str(op.dtype).split(".")[-1]


class PreparedThdLaunch:
    """The bf16 / f16 THD plan's launch, prepared once (see the module docstring).

    Built by the lowering after ``api.compile()``; ``None`` (caller falls back to
    the tensor-argument path) when the artifact has no positional tvm-ffi entry
    or the plan needs something this path does not cover yet (paged K/V,
    quantized operands, synthesized lengths, bias, gate).
    """

    def __init__(self, api, binding, *, scale_softmax: Optional[float]):
        km = api._k_mod
        compiled = api._compiled_kernel
        self._owner = compiled  # keeps the loaded module / JIT object alive with the plan
        raw = positional_entry(compiled)
        if raw is None:
            raise NotImplementedError("the compiled artifact exposes no positional tvm-ffi entry")
        self._fn = raw
        # Positional ABI of the C entry: the host's parameters minus the Constexpr ones (baked), stream last.
        order = [n for n, p in inspect.signature(km._host).parameters.items() if "Constexpr" not in str(p.annotation)]
        if order[-1] != "stream":
            raise NotImplementedError(f"{km.__name__}: the host entry does not end with the stream parameter: {order[-3:]}")
        self._index: Dict[str, int] = {n: i for i, n in enumerate(order)}
        wrapper_sig = getattr(compiled, "_kwargs_wrapper", None) or compiled
        try:
            seen = list(inspect.signature(wrapper_sig).parameters)
        except (TypeError, ValueError):
            seen = None
        if seen is not None and seen != order:
            raise NotImplementedError(f"{km.__name__}: positional ABI {order} disagrees with the compiled wrapper {seen}")

        plan = api._thd_plan()
        self._api = api
        self._plan = plan
        self._b, self._qh, self._kh = api.batch_size, api.h_q, api.h_kv
        self._d_qk, self._d_v = api.head_dim_qk, api.head_dim_v
        self._scale_log2 = float(api.scale_softmax if scale_softmax is None or scale_softmax == 0.0 else scale_softmax) * math.log2(math.e)
        self._expect = {
            "q": _dtype_name(api.q_desc),
            "k": _dtype_name(api.k_desc),
            "v": _dtype_name(api.v_desc),
            "o": _dtype_name(api.o_desc),
        }
        self._has_lse = api.lse_desc is not None
        self._lse_padded = bool(api.thd_stats_padded)
        self._lse_head_major = bool(api.thd_stats_head_major)
        self._lse_head_stride = int(api.thd_stats_head_stride or 0)
        self._lse_stride = tuple(int(x) for x in api._lse_stride) if self._lse_padded else None
        self._s_q_max = int(api.s_q_max)
        self._required_ws = int(plan.scratch_bytes)
        # The normalized pack re-describes an operand with the GRAPH's geometry when the caller's buffer covers the
        # declared bytes (``pack.graph_described``); a ragged declaration's span says nothing about the buffer, so the
        # token capacity of such an operand is the declared packed total, which the graph must therefore carry.
        self._total_q = None if plan.total_q is None else int(plan.total_q)
        self._total_kv = None if plan.total_kv is None else int(plan.total_kv)
        self._neg_inf = _buffers.init_word("fp32", float("-inf"))

        # variant-pack operands this plan binds, by IR tensor uid; indices resolved on first execute
        uids = {
            "q": binding.q.get_uid(),
            "k": binding.k.get_uid(),
            "v": binding.v.get_uid(),
            "o": binding.o.get_uid(),
            "q_lens": (binding.cu_seq_len_q if api.cu_seq_q_lens else binding.seq_len_q).get_uid(),
            "kv_lens": (binding.cu_seq_len_kv if api.cu_seq_kv_lens else binding.seq_len_kv).get_uid(),
        }
        if self._has_lse:
            uids["lse"] = binding.stats.get_uid()
        if api.has_sink:
            uids["sinks"] = binding.sink_token.get_uid()
        self._uid_names = list(uids)
        self._slot = {n: i for i, n in enumerate(self._uid_names)}
        self._uids = [uids[n] for n in self._uid_names]
        self._indices: Optional[List[int]] = None

        # the read-only template: plan constants; per-call slots left None
        t: List[Any] = [None] * len(order)
        idx = self._index

        def put(name, value):
            if name in idx:
                t[idx[name]] = value

        q, k, v, o = plan.q, plan.k, plan.v, plan.o  # (h, d, token_stride, head_stride, elem_stride, row_span)
        self._q_st, self._k_st, self._v_st, self._o_st = (q[2], q[2], q[3]), (k[2], k[2], k[3]), (v[2], v[2], v[3]), (o[2], o[2], o[3])
        put("q_strides", self._q_st)
        put("k_strides", self._k_st)
        put("v_strides", self._v_st)
        put("o_strides", self._o_st)
        if self._lse_padded:
            put("lse_strides", self._lse_stride)
            put("lse_ext", self._s_q_max)
        elif self._lse_head_major and self._lse_head_stride:
            put("lse_strides", (0, 0, 0))
            put("lse_ext", self._lse_head_stride)
        else:
            put("lse_strides", (0, 0, 0))
            put("lse_ext", 0)  # compact head-major: the token capacity, written per call
        put("scale_softmax_log2", self._scale_log2)
        put("n_thd_units", int(plan.units))
        put("seq_q_lens_addr", 0)
        put("thd_lens_form", int(plan.lens_form))
        put("o_partial_ptr", None)
        put("block_table_ptr", None)
        put("block_table_v_ptr", None)
        put("table_strides", (0, 0))
        put("n_pages", 0)
        self._template = t
        self._i_ptr = {n: idx[n + "_ptr"] for n in ("q", "k", "v", "o", "lse", "sinks", "meta", "o_desc")}
        self._i_q_lens, self._i_kv_lens = idx["thd_q_lens_ptr"], idx["thd_kv_lens_ptr"]
        self._i_problem, self._i_k_st, self._i_v_st, self._i_lse_ext, self._i_stream = (
            idx["problem_size"],
            idx["k_strides"],
            idx["v_strides"],
            idx["lse_ext"],
            idx["stream"],
        )
        self._stub_v = None  # (kh * d_v) zeros for the all-KV-zero clamp, allocated on first need
        self._sinks_dummy = None

    # -- per call ---------------------------------------------------------------------------

    def _resolve(self, pack) -> List[int]:
        try:
            self._indices = [pack.index_of(u) for u in self._uids]
        except KeyError as exc:
            raise ValueError(f"cudnn.sdpa: tensor uid {exc} is bound by the plan but is not an operand of this graph") from exc
        return self._indices

    def _lens(self, op, name: str, n: int) -> int:
        if _dtype_name(op) != "int32":
            raise ValueError(f"{name} must be int32; got {op.dtype}")
        if op.numel() != n:
            raise ValueError(f"{name} must have {n} elements; got {op.numel()}")
        if not _buffers.is_contiguous(tuple(op.shape), tuple(op.stride())):  # the graph may describe it as (n, 1, 1, 1)
            raise ValueError(f"{name} must be contiguous (read as a flat ({n},) operand)")
        return op.data_ptr()

    def _operand(self, op, name: str, pack, index: int) -> int:
        if _dtype_name(op) != self._expect[name]:
            raise ValueError(f"{name}: runtime buffer dtype {op.dtype} does not match its declaration ({self._expect[name]})")
        dev_type, dev_id = pack.observed_device(index)
        if dev_type != -1 and (dev_type != 2 or dev_id != pack.device):  # kDLCUDA == 2
            raise ValueError(f"{name}: runtime buffer is on DLPack device ({dev_type}, {dev_id}); this plan executes on CUDA device {pack.device}")
        ptr = op.data_ptr()
        if ptr % _ALIGN_TMA != 0:
            raise ValueError(
                f"{name}: runtime buffer base address must be 16-byte aligned (TMA global-address rule); got data_ptr() % 16 == {ptr % _ALIGN_TMA}"
            )
        return ptr

    def _seed_padded(self, lse_ptr: int, stream_int: int) -> None:
        shape = (self._b, self._qh, self._s_q_max)
        if _buffers.is_contiguous(shape, self._lse_stride):
            _buffers.fill_word_async(lse_ptr, math.prod(shape), self._neg_inf, stream_int)
        else:
            _buffers.fill_word_strided_async(lse_ptr, shape, self._lse_stride, 4, self._neg_inf, stream_int)

    def execute(self, pack, workspace_ptr: int, stream, stream_int: int) -> None:
        indices = self._indices or self._resolve(pack)
        ops = dict(zip(self._uid_names, pack.operands(indices)))

        def cap(name, decl):
            # the producer's span, never the effective (graph-described / overridden) geometry
            span = pack.observed_span(indices[self._slot[name]])
            if span < 0:
                raise ValueError(f"cudnn.sdpa: {name} was passed as a bare address; a ragged operand needs a sized buffer")
            row, ts = decl[5], decl[2]
            return 0 if span < row else (span - row) // ts + 1

        plan = self._plan
        frame = list(self._template)
        q, k, v, o = ops["q"], ops["k"], ops["v"], ops["o"]
        lse = ops.get("lse")
        frame[self._i_ptr["q"]] = self._operand(q, "q", pack, indices[self._slot["q"]])
        frame[self._i_ptr["k"]] = self._operand(k, "k", pack, indices[self._slot["k"]])
        frame[self._i_ptr["v"]] = self._operand(v, "v", pack, indices[self._slot["v"]])
        frame[self._i_ptr["o"]] = self._operand(o, "o", pack, indices[self._slot["o"]])
        frame[self._i_q_lens] = self._lens(ops["q_lens"], "cu_seq_len_q" if plan.lens_form & 1 else "seq_q_lens", plan.n_q_lens)
        frame[self._i_kv_lens] = self._lens(ops["kv_lens"], "cu_seq_len_kv" if plan.lens_form & 2 else "seq_kv_lens", plan.n_kv_lens)

        lse_ptr = None
        lse_cap = None
        if lse is not None:
            if _dtype_name(lse) != "float32":
                raise ValueError(f"lse_tensor must be float32; got {lse.dtype}")
            lse_ptr = lse.data_ptr()
            if lse_ptr % _ALIGN_F32 != 0:
                raise ValueError("lse_tensor must be 4-byte aligned")
            if self._lse_padded:
                expected = self._b * self._qh * self._s_q_max
                if lse.numel() != expected:
                    raise ValueError(f"padded lse_tensor must have B*H_q*S_q_max = {expected} elements; got {lse.numel()}")
            elif self._lse_head_major and self._lse_head_stride:
                if lse.numel() < self._qh * self._lse_head_stride:
                    raise ValueError(f"head-major lse_tensor must hold H_q*head_stride = {self._qh * self._lse_head_stride} elements; got {lse.numel()}")
            else:
                span = pack.observed_span(indices[self._slot["lse"]])
                lse_cap = (lse.numel() if span < 0 else span) // self._qh
        frame[self._i_ptr["lse"]] = lse_ptr

        t_q = min(cap("q", plan.q), cap("o", plan.o))
        if self._total_q is not None:
            t_q = min(t_q, self._total_q)
        if lse_cap is not None:
            t_q = min(t_q, lse_cap)
        if t_q == 0:
            if lse_ptr is not None and self._lse_padded:
                self._seed_padded(lse_ptr, stream_int)
            return  # no addressable Q token: nothing to launch
        t_kv = min(cap("k", plan.k), cap("v", plan.v))
        if self._total_kv is not None:
            t_kv = min(t_kv, self._total_kv)
        if t_kv == 0:
            # all-KV-zero clamp: one packed row of K aliases Q's storage, V a zero stub; the kernel reads no K/V row
            kh, d_qk, d_v = self._kh, self._d_qk, self._d_v
            t_kv = 1
            frame[self._i_ptr["k"]] = frame[self._i_ptr["q"]]
            frame[self._i_k_st] = (kh * d_qk, kh * d_qk, d_qk)
            frame[self._i_ptr["v"]] = self._v_stub(pack, stream)
            frame[self._i_v_st] = (kh * d_v, kh * d_v, d_v)
        if self._has_lse and self._lse_head_major and not self._lse_head_stride:
            frame[self._i_lse_ext] = t_q
        frame[self._i_problem] = (self._b, self._qh, self._kh, t_q, t_kv, 0)

        sinks = ops.get("sinks")
        if sinks is not None:
            if _dtype_name(sinks) != "float32" or sinks.numel() != self._qh:
                raise ValueError(f"sinks must be a ({self._qh},) float32 tensor; got dtype={sinks.dtype} numel={sinks.numel()}")
            frame[self._i_ptr["sinks"]] = sinks.data_ptr()
        else:
            frame[self._i_ptr["sinks"]] = self._sinks(pack, stream)

        if workspace_ptr % _ALIGN_TMA != 0:
            raise ValueError(f"cudnn.sdpa: the workspace must be 16-byte aligned; got 0x{workspace_ptr:x}")
        frame[self._i_ptr["meta"]] = workspace_ptr
        frame[self._i_ptr["o_desc"]] = workspace_ptr + plan.off_o_desc
        frame[self._i_stream] = stream

        if lse_ptr is not None and self._lse_padded:
            self._seed_padded(lse_ptr, stream_int)  # declared per-call operation: rows past each length read -inf
        self._fn(*frame)

    # -- read-only dummies, allocated once on the launch stream --------------------------------

    def _sinks(self, pack, stream) -> int:
        if self._sinks_dummy is None:
            import torch

            dev = torch.device("cuda", pack.device)
            self._sinks_dummy = self._api._dummy("sinks", dev, lambda: torch.zeros(self._qh, dtype=torch.float32, device=dev), stream=stream)
        return self._sinks_dummy.data_ptr()

    def _v_stub(self, pack, stream) -> int:
        if self._stub_v is None:
            import torch

            dev = torch.device("cuda", pack.device)
            kh, d_v = self._kh, self._d_v
            dt = getattr(torch, self._expect["v"])
            self._stub_v = self._api._dummy(f"thd_v_stub_{d_v}", dev, lambda: torch.zeros(kh * d_v, dtype=dt, device=dev), stream=stream)
        return self._stub_v.data_ptr()
