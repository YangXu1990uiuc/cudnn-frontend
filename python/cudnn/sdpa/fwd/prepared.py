# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""The SM100 THD (packed / ragged) f16 forward launch, prepared once and bound per call.

Three owners, one implementation each:

* **Observation** — :class:`BufferFacts`: what the caller's buffer is (address, dtype, device,
  the element span the producer guarantees, shape / strides). Read from the graph's normalized
  ``VariantPack`` (:func:`facts_of_pack`) or from a torch tensor handed to the standalone
  adapter (:func:`facts_of_tensor`). Nothing downstream looks at a buffer object again.
* **Semantics** — :class:`ThdLaunchSpec`, built by the adapter after ``compile()``: the
  positional argument template of the explicit host entry with every plan constant filled,
  the argument slot of every runtime field, the per-operand rules (dtype, alignment, extent),
  the capacity formulas, the workspace regions, the declared per-call operation (padded-Stats
  ``-inf`` seed) and the read-only dummies it owns.
* **Binding** — :func:`bind_thd`: applies the spec's rules to this call's facts and returns an
  independent argument frame (or None when no Q token is addressable). Lookups, integer
  arithmetic and writes; no ``cute`` objects, no torch views, no device allocation, no compile.

The graph plan (:class:`PreparedThdLaunch`) and the adapter's ``execute()`` both go through
``bind_thd`` and the artifact's positional tvm-ffi entry.
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from cudnn.frost import buffers as _buffers
from cudnn.frost.compiled_cache import positional_entry

_ALIGN_TMA = 16
_ALIGN_F32 = 4
_KDLCUDA = 2
_DTYPE_BY_CODE = {(code, bits): name for name, (code, bits) in _buffers.DTYPES.items()}


class BufferFacts(NamedTuple):
    """What a caller buffer is, observed once (see the module docstring)."""

    ptr: int
    dtype: str  # bare name ("bfloat16"); "" when unknown
    device: Tuple[int, int]  # DLPack (device_type, device_id); (-1, -1) unknown
    span: int  # element span the producer guarantees, in the DECLARED element width; -1 unknown (a bare address)
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]

    @property
    def numel(self) -> int:
        n = 1
        for e in self.shape:
            n *= int(e)
        return n

    @property
    def contiguous(self) -> bool:
        return _buffers.is_contiguous(self.shape, self.strides)


def facts_of_tensor(t) -> Optional[BufferFacts]:
    """Facts of a torch tensor (the standalone adapter's operands); None for None."""
    if t is None:
        return None
    shape, strides = tuple(t.shape), tuple(t.stride())
    n = 1
    for e in shape:
        n *= int(e)
    span = n if (n == 0 or t.is_contiguous()) else 1 + sum((int(s) - 1) * int(st) for s, st in zip(shape, strides))
    dev = t.device
    device = (_KDLCUDA, int(dev.index if dev.index is not None else 0)) if dev.type == "cuda" else (-1, -1)
    return BufferFacts(t.data_ptr(), str(t.dtype).split(".")[-1], device, span, shape, strides)


def facts_of_pack(pack, index: int) -> BufferFacts:
    """Facts of variant-pack operand ``index``: the producer's observed span / device, the
    effective (graph-described, overridden) geometry; no operand object is built."""
    native = pack.native
    code_bits = tuple(native.dtype(index))
    nbytes = native.observed_bytes(index)
    width = max(1, (int(code_bits[1]) + 7) // 8) if len(code_bits) == 2 else 1
    return BufferFacts(
        native.pointer(index),
        _DTYPE_BY_CODE.get(code_bits, ""),
        tuple(native.observed_device(index)),
        -1 if nbytes < 0 else nbytes // width,  # producer bytes -> elements of the EFFECTIVE (declared) dtype
        tuple(native.shape(index)),
        tuple(native.stride(index)),
    )


class ThdLaunchSpec:
    """Plan-time facts of one THD f16 launch (built by :func:`build_thd_spec`); read-only after
    build except for the lazily allocated read-only dummies it owns."""

    __slots__ = (
        "fn",
        "owner",
        "order",
        "index",
        "template",
        "b",
        "qh",
        "kh",
        "d_qk",
        "d_v",
        "paged",
        "page_size",
        "decl",
        "expect",
        "has_lse",
        "has_sink",
        "lse_padded",
        "lse_head_major",
        "lse_head_stride",
        "lse_stride",
        "s_q_max",
        "total_q",
        "total_kv",
        "n_q_lens",
        "n_kv_lens",
        "lens_form",
        "off_o_desc",
        "scratch_bytes",
        "neg_inf",
        "device_index",
        "_dummies",
    )

    def frame(self) -> List[Any]:
        return list(self.template)

    def dummy(self, key: str) -> int:
        """Address of a zero-filled read-only device buffer owned by this spec; every one the frame can
        need is allocated and initialized at build (:func:`build_thd_spec`), never during execute."""
        return self._dummies[key].data_ptr()


def _zeroed_device_buffer(nbytes: int, device_index: int) -> "_buffers.DeviceBuffer":
    """A ``cuMemAlloc`` buffer zeroed synchronously (cuMemsetD32 + stream sync at build): ready for
    whatever stream later reads it."""
    from cuda.bindings import driver as _drv

    buf = _buffers.DeviceBuffer(int(nbytes), device_index)
    (err,) = _drv.cuMemsetD32(buf.data_ptr(), 0, (int(nbytes) + 3) // 4)
    if int(err) != 0:
        raise RuntimeError(f"cudnn.sdpa: cuMemsetD32 failed: {err}")
    (err,) = _drv.cuStreamSynchronize(_drv.CUstream(0))
    if int(err) != 0:
        raise RuntimeError(f"cudnn.sdpa: cuStreamSynchronize failed: {err}")
    return buf


def build_thd_spec(api, *, scale_softmax: Optional[float]) -> ThdLaunchSpec:
    """The adapter's plan-time facts as a :class:`ThdLaunchSpec`; NotImplementedError when the
    artifact has no positional entry or the host signature is not the expected shape."""
    km = api._k_mod
    compiled = api._compiled_kernel
    raw = positional_entry(compiled)
    if raw is None:
        raise NotImplementedError("the compiled artifact exposes no positional tvm-ffi entry")
    order = [n for n, p in inspect.signature(km._host).parameters.items() if "Constexpr" not in str(p.annotation)]
    if order[-1] != "stream":
        raise NotImplementedError(f"{km.__name__}: the host entry does not end with the stream parameter: {order[-3:]}")
    wrapper_sig = getattr(compiled, "_kwargs_wrapper", None) or compiled
    try:
        seen = list(inspect.signature(wrapper_sig).parameters)
    except (TypeError, ValueError):
        seen = None
    if seen is not None and seen != order:
        raise NotImplementedError(f"{km.__name__}: positional ABI {order} disagrees with the compiled wrapper {seen}")
    plan = api._thd_plan()
    s = ThdLaunchSpec()
    s.fn, s.owner, s.order = raw, compiled, order
    s.index = {n: i for i, n in enumerate(order)}
    s.b, s.qh, s.kh, s.d_qk, s.d_v = api.batch_size, api.h_q, api.h_kv, api.head_dim_qk, api.head_dim_v
    s.paged, s.page_size = bool(api.paged), int(api.paged_page_size or 0)
    s.decl = dict(q=plan.q, k=plan.k, v=plan.v, o=plan.o)  # (h, d, token_stride, head_stride, elem_stride, row_span)
    s.expect = {n: str(getattr(api, f"{n}_desc").dtype).split(".")[-1] for n in ("q", "k", "v", "o")}
    s.has_lse, s.has_sink = api.lse_desc is not None, bool(api.has_sink)
    s.lse_padded, s.lse_head_major = bool(api.thd_stats_padded), bool(api.thd_stats_head_major)
    s.lse_head_stride = int(api.thd_stats_head_stride or 0)
    s.lse_stride = tuple(int(x) for x in api._lse_stride) if s.lse_padded else None
    s.s_q_max = int(api.s_q_max)
    s.total_q = None if plan.total_q is None else int(plan.total_q)
    s.total_kv = None if plan.total_kv is None else int(plan.total_kv)
    s.n_q_lens, s.n_kv_lens, s.lens_form = int(plan.n_q_lens), int(plan.n_kv_lens), int(plan.lens_form)
    s.off_o_desc, s.scratch_bytes = int(plan.off_o_desc), int(plan.scratch_bytes)
    s.neg_inf = _buffers.init_word("fp32", float("-inf"))
    s.device_index = int(api.q_desc.device.index or 0)
    s._dummies = {}
    if not s.has_sink:
        s._dummies["sinks"] = _zeroed_device_buffer(s.qh * 4, s.device_index)
    if not s.paged:  # the all-KV-zero clamp's V stub: one (kh, d_v) row of zeros
        s._dummies["v_stub"] = _zeroed_device_buffer(s.kh * s.d_v * _buffers.DTYPE_ITEMSIZE[s.expect["v"]], s.device_index)
    scale = float(api.scale_softmax if scale_softmax is None or scale_softmax == 0.0 else scale_softmax)

    t: List[Any] = [None] * len(order)

    def put(name, value):
        if name in s.index:
            t[s.index[name]] = value

    q, k, v, o = plan.q, plan.k, plan.v, plan.o
    put("q_strides", (q[2], q[2], q[3]))
    put("o_strides", (o[2], o[2], o[3]))
    if not s.paged:
        put("k_strides", (k[2], k[2], k[3]))
        put("v_strides", (v[2], v[2], v[3]))
    if s.lse_padded:
        put("lse_strides", s.lse_stride)
        put("lse_ext", s.s_q_max)
    else:
        put("lse_strides", (0, 0, 0))
        put("lse_ext", s.lse_head_stride)  # compact head-major: the token capacity, written per call
    put("scale_softmax_log2", scale * math.log2(math.e))
    put("n_thd_units", int(plan.units))
    put("seq_q_lens_addr", 0)
    put("thd_lens_form", s.lens_form)
    put("o_partial_ptr", None)
    put("block_table_ptr", None)
    put("block_table_v_ptr", None)
    put("table_strides", (0, 0))
    put("n_pages", 0)
    s.template = t
    return s


def _check(cond: bool, msg: str) -> None:
    if cond:
        raise ValueError(f"cudnn.sdpa: {msg}")


def _capacity(f: BufferFacts, decl: tuple, name: str) -> int:
    _check(f.span < 0, f"{name} was passed as a bare address; a ragged operand needs a sized buffer")
    row, ts = decl[5], decl[2]
    return 0 if f.span < row else (f.span - row) // ts + 1


def bind_thd(spec: ThdLaunchSpec, facts: Dict[str, Optional[BufferFacts]], workspace_ptr: int, stream, stream_int: int) -> Optional[List[Any]]:
    """This call's argument frame for ``spec`` from the operands' facts (roles ``q k v o lse sinks
    q_lens kv_lens`` and, paged, ``block_table block_table_v``); None when no Q token is
    addressable. Runs the declared per-call operation (padded-Stats seed) on ``stream_int``."""
    ix = spec.index
    frame = spec.frame()

    def on_plan_device(name: str, f: BufferFacts) -> None:
        # one device rule for every bound role: a KNOWN producer device must be the plan's CUDA device
        _check(
            f.device[0] != -1 and f.device != (_KDLCUDA, spec.device_index),
            f"{name}: runtime buffer is on DLPack device {f.device}; this plan executes on CUDA device {spec.device_index}",
        )

    def operand(name: str, packed: bool = True) -> BufferFacts:
        f = facts.get(name)
        _check(f is None, f"{name} is required")
        _check(f.dtype != spec.expect[name], f"{name}: runtime buffer dtype {f.dtype} does not match its declaration ({spec.expect[name]})")
        on_plan_device(name, f)
        _check(
            f.ptr % _ALIGN_TMA != 0,
            f"{name}: runtime buffer base address must be 16-byte aligned (TMA global-address rule); got data_ptr() % 16 == {f.ptr % _ALIGN_TMA}",
        )
        if packed and f.numel > 0:
            # Effective geometry must be the plan's declared (token, head, elem) strides: the artifact
            # takes strides at runtime, but a THD stride override is not bound yet, so a different
            # layout is rejected here instead of silently running on the declared one.
            st = f.strides
            if len(st) == 4:  # the graph's (B, H, S, D) declaration
                eff = (int(st[2]), int(st[1]), int(st[3]))
            elif len(st) == 3:  # the caller's packed (T, H, D)
                eff = (int(st[0]), int(st[1]), int(st[2]))
            else:
                eff = None
            _check(eff is None, f"{name}: a THD operand is (T, H, D) or the graph's (B, H, S, D); got rank {len(st)}")
            decl = spec.decl[name]
            _check(
                eff != (decl[2], decl[3], decl[4]),
                f"{name}: runtime (token, head, elem) strides {eff} differ from the plan's declared {(decl[2], decl[3], decl[4])}; THD stride override is not supported",
            )
        return f

    def lens(name: str, n: int) -> int:
        f = facts.get(name)
        _check(f is None, f"{name} is required")
        on_plan_device(name, f)
        _check(f.dtype != "int32", f"{name} must be int32; got {f.dtype}")
        _check(f.numel != n, f"{name} must have {n} elements; got {f.numel}")
        _check(not f.contiguous, f"{name} must be contiguous (read as a flat ({n},) operand)")
        return f.ptr

    q, o = operand("q"), operand("o")
    k, v = operand("k", packed=not spec.paged), operand("v", packed=not spec.paged)
    frame[ix["q_ptr"]], frame[ix["k_ptr"]], frame[ix["v_ptr"]], frame[ix["o_ptr"]] = q.ptr, k.ptr, v.ptr, o.ptr
    frame[ix["thd_q_lens_ptr"]] = lens("q_lens", spec.n_q_lens)
    frame[ix["thd_kv_lens_ptr"]] = lens("kv_lens", spec.n_kv_lens)

    lse = facts.get("lse")
    lse_cap = None
    if spec.has_lse:
        _check(lse is None, "lse_tensor is required by this compiled specialization")
        on_plan_device("lse_tensor", lse)
        _check(lse.dtype != "float32", f"lse_tensor must be float32; got {lse.dtype}")
        _check(lse.ptr % _ALIGN_F32 != 0, "lse_tensor must be 4-byte aligned")
        if spec.lse_padded:
            expected = spec.b * spec.qh * spec.s_q_max
            _check(lse.numel != expected, f"padded lse_tensor must have B*H_q*S_q_max = {expected} elements; got {lse.numel}")
        elif spec.lse_head_major and spec.lse_head_stride:
            _check(
                lse.numel < spec.qh * spec.lse_head_stride,
                f"head-major lse_tensor must hold H_q*head_stride = {spec.qh * spec.lse_head_stride} elements; got {lse.numel}",
            )
        else:
            lse_cap = (lse.numel if lse.span < 0 else lse.span) // spec.qh
        frame[ix["lse_ptr"]] = lse.ptr
    else:
        _check(lse is not None, "this specialization was compiled without a Stats output; construct the API without sample_lse")

    def seed_padded():
        shape = (spec.b, spec.qh, spec.s_q_max)
        if _buffers.is_contiguous(shape, spec.lse_stride):
            _buffers.fill_word_async(lse.ptr, math.prod(shape), spec.neg_inf, stream_int)
        else:
            _buffers.fill_word_strided_async(lse.ptr, shape, spec.lse_stride, 4, spec.neg_inf, stream_int)

    t_q = min(_capacity(q, spec.decl["q"], "q"), _capacity(o, spec.decl["o"], "o"))
    if spec.total_q is not None:
        t_q = min(t_q, spec.total_q)
    if lse_cap is not None:
        t_q = min(t_q, lse_cap)
    if t_q == 0:
        if spec.has_lse and spec.lse_padded:
            seed_padded()
        return None

    if spec.paged:
        # K/V are page pools (n_pages, page_size, KH, D) in the kernel's order: a permutation of the
        # container's (n_pages, KH, page_size, D) strides; SKV = max_pages * page_size
        bt, btv = facts.get("block_table"), facts.get("block_table_v")
        _check(bt is None or btv is None, "paged KV requires paged_attention_k_table / paged_attention_v_table buffers")
        _check(bt.dtype != "int32" or btv.dtype != "int32", "the page tables must be int32")
        on_plan_device("paged_attention_k_table", bt)
        on_plan_device("paged_attention_v_table", btv)
        _check(
            bt.span >= 0 and bt.span < (bt.shape[0] * bt.shape[2] if len(bt.shape) == 4 else bt.numel),
            "paged_attention_k_table is smaller than its declared (B, max_pages) extent",
        )
        if len(bt.shape) == 4:  # the graph's (B, 1, max_pages, 1) declaration
            table_shape = (bt.shape[0], bt.shape[2])
            table_strides = (bt.strides[0], bt.strides[2])
        else:
            table_shape, table_strides = tuple(bt.shape), tuple(bt.strides)
        _check(len(table_shape) != 2, f"the page table must be (B, max_pages); got {bt.shape}")
        t_kv = int(table_shape[1]) * spec.page_size
        frame[ix["k_strides"]] = (int(k.strides[0]), int(k.strides[2]), int(k.strides[1]))
        frame[ix["v_strides"]] = (int(v.strides[0]), int(v.strides[2]), int(v.strides[1]))
        frame[ix["block_table_ptr"]], frame[ix["block_table_v_ptr"]] = bt.ptr, btv.ptr
        frame[ix["table_strides"]] = (int(table_strides[0]), int(table_strides[1]))
        frame[ix["n_pages"]] = int(k.shape[0])
    else:
        t_kv = min(_capacity(k, spec.decl["k"], "k"), _capacity(v, spec.decl["v"], "v"))
        if spec.total_kv is not None:
            t_kv = min(t_kv, spec.total_kv)
        if t_kv == 0:
            # all-KV-zero clamp: one packed row of K aliases Q's storage, V a zero stub; the kernel reads no K/V row
            kh, d_qk, d_v = spec.kh, spec.d_qk, spec.d_v
            t_kv = 1
            frame[ix["k_ptr"]] = q.ptr
            frame[ix["k_strides"]] = (kh * d_qk, kh * d_qk, d_qk)
            frame[ix["v_ptr"]] = spec.dummy("v_stub")
            frame[ix["v_strides"]] = (kh * d_v, kh * d_v, d_v)
    if spec.has_lse and spec.lse_head_major and not spec.lse_head_stride:
        frame[ix["lse_ext"]] = t_q
    frame[ix["problem_size"]] = (spec.b, spec.qh, spec.kh, t_q, t_kv, 0)

    sinks = facts.get("sinks")
    if spec.has_sink:
        _check(sinks is None, "sinks is required by this compiled specialization")
        on_plan_device("sinks", sinks)
        _check(sinks.dtype != "float32" or sinks.numel != spec.qh or not sinks.contiguous, f"sinks must be a contiguous ({spec.qh},) float32 tensor")
        frame[ix["sinks_ptr"]] = sinks.ptr
    else:
        _check(sinks is not None, "this specialization was compiled without a sink; construct the API with has_sink")
        frame[ix["sinks_ptr"]] = spec.dummy("sinks")

    _check(workspace_ptr % _ALIGN_TMA != 0, f"the workspace must be 16-byte aligned; got 0x{workspace_ptr:x}")
    frame[ix["meta_ptr"]] = workspace_ptr
    frame[ix["o_desc_ptr"]] = workspace_ptr + spec.off_o_desc
    frame[ix["stream"]] = stream
    if spec.has_lse and spec.lse_padded:
        seed_padded()  # declared per-call operation: rows past each length read -inf
    return frame


class PreparedThdLaunch:
    """The graph plan's THD f16 launch: the spec plus this graph's operand uids."""

    def __init__(self, spec: ThdLaunchSpec, binding):
        self.spec = spec
        uids = {
            "q": binding.q.get_uid(),
            "k": binding.k.get_uid(),
            "v": binding.v.get_uid(),
            "o": binding.o.get_uid(),
            "q_lens": (binding.cu_seq_len_q if spec.lens_form & 1 else binding.seq_len_q).get_uid(),
            "kv_lens": (binding.cu_seq_len_kv if spec.lens_form & 2 else binding.seq_len_kv).get_uid(),
        }
        if spec.has_lse:
            uids["lse"] = binding.stats.get_uid()
        if spec.has_sink:
            uids["sinks"] = binding.sink_token.get_uid()
        if spec.paged:
            uids["block_table"] = binding.paged_k_table.get_uid()
            uids["block_table_v"] = binding.paged_v_table.get_uid()
        self._roles = list(uids)
        self._uids = [uids[r] for r in self._roles]
        self._indices: Optional[List[int]] = None

    def execute(self, pack, workspace_ptr: int, stream, stream_int: int) -> None:
        indices = self._indices
        if indices is None:
            try:
                indices = self._indices = [pack.index_of(u) for u in self._uids]
            except KeyError as exc:
                raise ValueError(f"cudnn.sdpa: tensor uid {exc} is bound by the plan but is not an operand of this graph") from exc
        facts = {role: facts_of_pack(pack, i) for role, i in zip(self._roles, indices)}
        frame = bind_thd(self.spec, facts, workspace_ptr, stream, stream_int)
        if frame is not None:
            self.spec.fn(*frame)
