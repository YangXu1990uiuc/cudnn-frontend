# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FROST SM100 D128 BF16 NVFP4 QAT fused backward kernel (cga1).

One CTA owns 128 KV rows and loops over Q tiles of 128. Per Q tile the MMA warp
issues five cta_group::1 MMAs: S = K·Qᵀ, dP = V·dOᵀ, dV += fq(P)·dO, dK += dS·Q
and dQ_part = dSᵀ·K. dV/dK accumulate in TMEM for the whole KV tile; the dQ
partial is staged as FP32 through the dS shared-memory slot and reduce-added
into an FP32 dQ accumulator with TMA (cp.reduce.async.bulk.tensor). With
``deterministic`` the reduces of one Q tile are released in KV-tile order
through a per-(batch, head, q-tile) semaphore, so the FP32 sum is bitwise
reproducible; without it the order is whatever the hardware does.

Because every CTA holds the whole [128 × 128] Q and dO tiles, one SMEM buffer
serves as the K-major B operand (S, dP) and as the MN-major B operand (dK, dV):
no second loads, unlike the 2-CTA kernels. SMEM: K 32K + V 32K + Q 2×32K +
dO 2×32K + dS 32K + stats 2K ≈ 226 KiB. TMEM: S/P 0..127, dP (and dQ_part)
128..255, dV 256..383, dK 384..511.

Layouts: Q/K/V BSHD (quantizer outputs); dO, dV, dK caller BHSD; dQ accumulator
FP32 BHSD; lse/delta [B, H, S] FP32 (raw; log2e and attn_scale folded in-kernel).
"""

from typing import Callable, Tuple

import cuda.bindings.driver as _cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm
from cutlass.experimental import primitives as prims
from cutlass.experimental.cuda import tensor_map as tmap
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer
from dataclasses import dataclass
from typing import NamedTuple

from cudnn.frost.tile_dsl.barrier import PipelineState, advance, MBarrier, Producer, wait, arrive_expect_tx
from cudnn.frost.tile_dsl.scheduler import Sched, read_tile_id_arrive
from cudnn.frost.tile_dsl.mma import mma_ss, mma_ts
from cudnn.frost.tile_dsl.tma import tma_load_tile, tma_store_tile, tma_store_commit, tma_store_wait
from cudnn.frost.tile_dsl.handles import MmaDesc, SmemTile, GmemTileTma
from cudnn.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from cudnn.frost.tile_dsl.pointwise import tmem_load_tile

from ._frost_kernel import _fake_quant_p_many

# ============================================================================
# Config — fixed: BF16, D128, dense, cga1, 12 warps.
# ============================================================================


@dataclass(frozen=True)
class FusedCfg:
    TILE_KV: int = 128  # kv rows per CTA
    TILE_Q: int = 128  # q per iteration
    D: int = 128
    BPE: int = 2
    STAGES_Q: int = 2
    STAGES_DO: int = 2
    TILE_K_HW: int = 16
    SWZ_BYTES: int = 128
    SOFTMAX_WARPGROUPS: int = 2
    SOFTMAX_WG_WARPS: int = 4
    SCHEDULER_STAGES: int = 2
    STATS_STAGES: int = 2
    # setmaxnreg: 8*(S-168) == 4*(168-R) around the compiled 168 for 12 warps.
    SOFTMAX_REGS: int = 216
    OTHER_REGS: int = 72
    THREADS_PER_CTA: int = 12 * 32
    MMA_WARP_ID: int = 8
    TMALDG_WARP_ID: int = 9
    TMASTG_WARP_ID: int = 10
    SCHED_WARP_ID: int = 11
    SOFTMAX_LANES: int = 256


CFG = FusedCfg()
assert 8 * (CFG.SOFTMAX_REGS - 168) == 4 * (168 - CFG.OTHER_REGS), "setmaxnreg inc/dec must balance around 168"

STORAGE_DTYPE = cutlass.BFloat16
OUT_STORAGE_DTYPE = cutlass.BFloat16
MMA_KIND = nvvm.Tcgen05MMAKind.F16
CTA_GROUP_KIND = nvvm.CTAGroup.CTA_1
CGA_SIZE = 1
_LOG2E = 1.4426950408889634

# --- SMEM geometry (elements) ---------------------------------------------
GRANU = CFG.SWZ_BYTES // CFG.BPE  # 64 elems per 128 B swizzle row
tileElems = CFG.TILE_KV * CFG.D  # every operand tile is [128 × 128]
SMEM_LAYOUT_SWZ128 = 2
STRIDE_BYTE_OFFSET = 8 * CFG.SWZ_BYTES
LEADING_BYTE_OFFSET_MN = 128 * CFG.SWZ_BYTES  # MN-major view: 64-col block (slab) stride
SLAB_ELEMS = 128 * GRANU  # one [128 rows × 64 cols] swizzled slab
P_SMEM_SWIZZLE = cutlass.Swizzle(3, 4, 3)
TMA_ITERS = CFG.D // GRANU  # 2 slabs per tile
tileTmaBytes = tileElems * CFG.BPE
_SMX_CHUNK = CFG.TILE_Q // CFG.SOFTMAX_WARPGROUPS  # 64 q cols per wg
_EPI_CHUNK = CFG.D // CFG.SOFTMAX_WARPGROUPS  # 64 d cols per wg (dV/dK epilogue)
DQ_HALVES = 2
DQ_STG_COLS = 32  # fp32 per lane per half = 128 B (one swizzle row)
DQ_SLAB_ELEMS = 128 * DQ_STG_COLS  # 4096 fp32 = 16 KiB per wg slab
assert DQ_HALVES * DQ_STG_COLS * CFG.SOFTMAX_WARPGROUPS == CFG.D
assert DQ_SLAB_ELEMS * CFG.SOFTMAX_WARPGROUPS * 4 == tileElems * CFG.BPE, "dQ staging must fit the dS slot"

STATS_LSE_OFF = 0
STATS_DOT_OFF = CFG.TILE_Q
STATS_SLOT_ELEMS = 2 * CFG.TILE_Q


@dataclass(frozen=True)
class TmemLayout:
    TOTAL_COLS: int = 512
    S_OFF: int = 0
    P_OFF: int = 32  # bf16 P inside the S region: wg0 [32..63], wg1 [64..95]
    dP_OFF: int = 128  # dP, then dQ_part
    dV_OFF: int = 256
    dK_OFF: int = 384


LAYOUT = TmemLayout()
assert LAYOUT.P_OFF + (CFG.TILE_Q * CFG.BPE) // 4 <= LAYOUT.S_OFF + 128

ONE_LANE = 1
ONE_WARP = 32
MMA_COMMIT_ARRIVES = 1
SOFTMAX_LANES = CFG.SOFTMAX_LANES
# 8 softmax + MMA + TMALDG + TMASTG loop and arrive once each per tile (cga1: local arrive).
READ_TILE_ARRIVERS_TOT = 11


class Bars(NamedTuple):
    mb_q_full: object
    mb_q_empty: object
    mb_do_full: object
    mb_do_empty: object
    mb_k_full: object
    mb_k_empty: object
    mb_v_full: object
    mb_v_empty: object
    mb_s_acc_full: object  # MMA(S) -> softmax
    mb_p_ready: object  # softmax -> MMA(dV): bf16 P in TMEM
    mb_dp_full: object  # MMA(dP) -> softmax
    mb_dp_empty: object  # softmax -> MMA(dP[i+1]): dP region drained (after dQ_part readout)
    mb_ds_ready: object  # softmax -> MMA(dK, dQ): dS in SMEM
    mb_dsq_done: object  # MMA commit after dK & dQ_part: dS slot free, dQ_part in TMEM
    mb_dq_stg_full: object  # softmax -> TMASTG: one FP32 half staged
    mb_dq_stg_empty: object  # TMASTG -> softmax: reduce finished reading the slot
    mb_stats_full: object
    mb_stats_empty: object
    mb_dvk_ready: object  # MMA commit after the tile's last dV/dK MMA
    mb_acc_empty: object  # softmax -> MMA: dV/dK TMEM read out
    mb_dv_stg_full: object
    mb_dv_stg_empty: object
    mb_dk_stg_full: object
    mb_dk_stg_empty: object
    mb_tmem_dealloc: object


def _make_bars():
    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    T, C, L = Producer.TMA_LOAD, Producer.MMA_COMMIT, Producer.THREAD
    return Bars(
        mb_q_full=MBarrier(_alloc(CFG.STAGES_Q), stages=CFG.STAGES_Q, init_count=ONE_LANE, producer=T),
        mb_q_empty=MBarrier(_alloc(CFG.STAGES_Q), stages=CFG.STAGES_Q, init_count=MMA_COMMIT_ARRIVES, producer=C),
        mb_do_full=MBarrier(_alloc(CFG.STAGES_DO), stages=CFG.STAGES_DO, init_count=ONE_LANE, producer=T),
        mb_do_empty=MBarrier(_alloc(CFG.STAGES_DO), stages=CFG.STAGES_DO, init_count=MMA_COMMIT_ARRIVES, producer=C),
        mb_k_full=MBarrier(_alloc(1), stages=1, init_count=ONE_LANE, producer=T),
        mb_k_empty=MBarrier(_alloc(1), stages=1, init_count=MMA_COMMIT_ARRIVES, producer=C),
        mb_v_full=MBarrier(_alloc(1), stages=1, init_count=ONE_LANE, producer=T),
        mb_v_empty=MBarrier(_alloc(1), stages=1, init_count=MMA_COMMIT_ARRIVES, producer=C),
        mb_s_acc_full=MBarrier(_alloc(1), stages=1, init_count=MMA_COMMIT_ARRIVES, producer=C),
        mb_p_ready=MBarrier(_alloc(1), stages=1, init_count=SOFTMAX_LANES, producer=L),
        mb_dp_full=MBarrier(_alloc(1), stages=1, init_count=MMA_COMMIT_ARRIVES, producer=C),
        mb_dp_empty=MBarrier(_alloc(1), stages=1, init_count=SOFTMAX_LANES, producer=L),
        mb_ds_ready=MBarrier(_alloc(1), stages=1, init_count=SOFTMAX_LANES, producer=L),
        mb_dsq_done=MBarrier(_alloc(1), stages=1, init_count=MMA_COMMIT_ARRIVES, producer=C),
        mb_dq_stg_full=MBarrier(_alloc(1), stages=1, init_count=SOFTMAX_LANES, producer=L),
        mb_dq_stg_empty=MBarrier(_alloc(1), stages=1, init_count=ONE_LANE, producer=L),
        mb_stats_full=MBarrier(_alloc(CFG.STATS_STAGES), stages=CFG.STATS_STAGES, init_count=ONE_WARP, producer=L),
        mb_stats_empty=MBarrier(_alloc(CFG.STATS_STAGES), stages=CFG.STATS_STAGES, init_count=SOFTMAX_LANES, producer=L),
        mb_dvk_ready=MBarrier(_alloc(1), stages=1, init_count=MMA_COMMIT_ARRIVES, producer=C),
        mb_acc_empty=MBarrier(_alloc(1), stages=1, init_count=SOFTMAX_LANES, producer=L),
        mb_dv_stg_full=MBarrier(_alloc(1), stages=1, init_count=SOFTMAX_LANES, producer=L),
        mb_dv_stg_empty=MBarrier(_alloc(1), stages=1, init_count=ONE_LANE, producer=L),
        mb_dk_stg_full=MBarrier(_alloc(1), stages=1, init_count=SOFTMAX_LANES, producer=L),
        mb_dk_stg_empty=MBarrier(_alloc(1), stages=1, init_count=ONE_LANE, producer=L),
        mb_tmem_dealloc=MBarrier(_alloc(1), stages=1, init_count=SOFTMAX_LANES, producer=L),
    )


def _smem_tile(base, stages: int, mn_major: bool):
    """[128 × 128] bf16 tile as 2 swizzled [128 × 64] slabs; MN-major views set the slab stride as LBO."""
    return SmemTile(
        desc_version=0,
        base=base,
        elems_per_stage=tileElems,
        stages=stages,
        leading_byte_offset=LEADING_BYTE_OFFSET_MN if mn_major else 0,
        stride_byte_offset=STRIDE_BYTE_OFFSET,
        layout=SMEM_LAYOUT_SWZ128,
        tma_loads_per_tile=TMA_ITERS,
        tma_granu_elems=GRANU,
        tma_subtile_stride_elems=SLAB_ELEMS,
    )


# ============================================================================
# Kernel entry
# ============================================================================


@cute.kernel
def _kernel(
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_do_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_v_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_dv_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_dk_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_dq_desc: cutlass.GridConstant[tmap.TensorMap],  # FP32 accumulator (reduce-add)
    lse_tensor: cute.Tensor,  # [B, H, S_q] FP32 natural log
    delta_tensor: cute.Tensor,  # [B, H, S_q] FP32 raw rowsum(O·dO)
    sem_tensor: cute.Tensor,  # [B, H, S_q/128] Int32, caller-zeroed (deterministic only)
    seqlen_q: cutlass.Int32,
    seqlen_kv: cutlass.Int32,
    n_qh: cutlass.Int32,
    n_batch: cutlass.Int32,
    qh_per_kh: cutlass.Int32,
    attn_scale: cutlass.Float32,
    attn_scale_log2e: cutlass.Float32,
    head_base: cutlass.Int32,
    deterministic: cutlass.Constexpr[bool],
) -> None:
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]

    # --- SMEM: K 32K + V 32K + Q 2×32K + dO 2×32K + dS 32K + stats 2K ---------
    sK_raw = cutlass.Array(STORAGE_DTYPE, tileElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sV_raw = cutlass.Array(STORAGE_DTYPE, tileElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sQ_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_Q * tileElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sdO_raw = cutlass.Array(STORAGE_DTYPE, CFG.STAGES_DO * tileElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sdS_raw = cutlass.Array(STORAGE_DTYPE, tileElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sStats_raw = cutlass.Array(cutlass.Float32, CFG.STATS_STAGES * STATS_SLOT_ELEMS, alignment=1024, space=cutlass.AddressSpace.smem)
    # Epilogue staging aliases dead operands: dV over K, dK over V (TMALDG waits the stores before reloading).
    sdV_raw = cutlass.Array(sK_raw.data_ptr(), tileElems, dtype=OUT_STORAGE_DTYPE)
    sdK_raw = cutlass.Array(sV_raw.data_ptr(), tileElems, dtype=OUT_STORAGE_DTYPE)

    sK = _smem_tile(sK_raw, 1, False)  # A of S (M=kv, K=d)
    sK_T = _smem_tile(sK_raw, 1, True)  # B of dQ (N=d, K=kv)
    sV = _smem_tile(sV_raw, 1, False)  # A of dP
    sQ = _smem_tile(sQ_raw, CFG.STAGES_Q, False)  # B of S (N=q, K=d)
    sQ_T = _smem_tile(sQ_raw, CFG.STAGES_Q, True)  # B of dK (N=d, K=q)
    sdO = _smem_tile(sdO_raw, CFG.STAGES_DO, False)  # B of dP
    sdO_T = _smem_tile(sdO_raw, CFG.STAGES_DO, True)  # B of dV (N=d, K=q)
    sdS = _smem_tile(sdS_raw, 1, False)  # A of dK (M=kv, K=q)
    sdS_T = _smem_tile(sdS_raw, 1, True)  # A of dQ (M=q, K=kv)
    sdV = SmemTile(
        desc_version=0,
        base=sdV_raw,
        elems_per_stage=tileElems,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=SMEM_LAYOUT_SWZ128,
        tma_loads_per_tile=TMA_ITERS,
        tma_granu_elems=GRANU,
        tma_subtile_stride_elems=SLAB_ELEMS,
    )
    sdK = SmemTile(
        desc_version=0,
        base=sdK_raw,
        elems_per_stage=tileElems,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=SMEM_LAYOUT_SWZ128,
        tma_loads_per_tile=TMA_ITERS,
        tma_granu_elems=GRANU,
        tma_subtile_stride_elems=SLAB_ELEMS,
    )

    bars = _make_bars()
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)
    sched = Sched(
        **{
            "mb_scheduler": cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "mb_read_tile_id": cutlass.Array(cutlass.Int64, CFG.SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "tile_id_smem": cutlass.Array(cutlass.Int32, CFG.SCHEDULER_STAGES * 8, alignment=16, space=cutlass.AddressSpace.smem),
            "bidx_init": bidx,
            "bidy_init": bidy,
            "bidz_init": bidz,
        }
    )

    if warp_idx == 0:
        if nvvm.elect_sync():
            for s in cutlass.range_constexpr(CFG.STAGES_Q):
                bars.mb_q_full[s].init()
                bars.mb_q_empty[s].init()
            for s in cutlass.range_constexpr(CFG.STAGES_DO):
                bars.mb_do_full[s].init()
                bars.mb_do_empty[s].init()
            bars.mb_k_full.init()
            bars.mb_k_empty.init()
            bars.mb_v_full.init()
            bars.mb_v_empty.init()
            bars.mb_s_acc_full.init()
            bars.mb_p_ready.init()
            bars.mb_dp_full.init()
            bars.mb_dp_empty.init()
            bars.mb_ds_ready.init()
            bars.mb_dsq_done.init()
            bars.mb_dq_stg_full.init()
            bars.mb_dq_stg_empty.init()
            for s in cutlass.range_constexpr(CFG.STATS_STAGES):
                bars.mb_stats_full[s].init()
                bars.mb_stats_empty[s].init()
            bars.mb_dvk_ready.init()
            bars.mb_acc_empty.init()
            bars.mb_dv_stg_full.init()
            bars.mb_dv_stg_empty.init()
            bars.mb_dk_stg_full.init()
            bars.mb_dk_stg_empty.init()
            bars.mb_tmem_dealloc.init()
            for s in range(CFG.SCHEDULER_STAGES):
                nvvm.mbarrier_init(sched.mb_scheduler.subview(s), ONE_LANE)
                nvvm.mbarrier_init(sched.mb_read_tile_id.subview(s), READ_TILE_ARRIVERS_TOT)

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    if warp_idx < cutlass.Int32(CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS):
        nvvm.setmaxregister(CFG.SOFTMAX_REGS, nvvm.SetMaxRegisterAction.INCREASE)
        _softmax_warp_group(warp_idx, tmem_ptr_i32, bars, sched, sdS_raw, sStats_raw, sdV_raw, sdK_raw, seqlen_q, seqlen_kv, attn_scale, attn_scale_log2e)
    elif warp_idx == cutlass.Int32(CFG.MMA_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        _mma_warp(sQ, sQ_T, sdO, sdO_T, sK, sK_T, sV, sdS, sdS_T, tmem_ptr_i32, bars, sched, seqlen_q, seqlen_kv)
    elif warp_idx == cutlass.Int32(CFG.TMALDG_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.prefetch_tensormap(tma_q_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_do_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_desc.get_ptr())
        _tmaldg_warp(tma_q_desc, tma_do_desc, tma_k_desc, tma_v_desc, sQ, sdO, sK, sV, bars, sched, seqlen_q, seqlen_kv, qh_per_kh, head_base)
    elif warp_idx == cutlass.Int32(CFG.TMASTG_WARP_ID):
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        nvvm.prefetch_tensormap(tma_dv_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_dk_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_dq_desc.get_ptr())
        _tmastg_warp(tma_dv_desc, tma_dk_desc, tma_dq_desc, sdV, sdK, sdS_raw, sem_tensor, bars, sched, seqlen_q, seqlen_kv, n_qh, head_base, deterministic)
    else:
        nvvm.setmaxregister(CFG.OTHER_REGS, nvvm.SetMaxRegisterAction.DECREASE)
        _scheduler_stats_warp(sched, bars, sStats_raw, lse_tensor, delta_tensor, attn_scale, seqlen_q, head_base)


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


# ============================================================================
# Tile decode (cga1, natural 3-D grid: x = kv tile, y = head, z = batch)
# ============================================================================


@cute.jit
def _boot_tile(sched):
    return sched.bidx_init, sched.bidy_init, sched.bidz_init


@cute.jit
def _decode_tile_payload(sched, sched_idx):
    """try_cancel payload = [blockIdx.x, packed(head | batch << 16), valid, 0]."""
    t0 = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_idx * cutlass.Int32(8) + cutlass.Int32(0))).load())
    t1 = cute.arch.make_warp_uniform((sched.tile_id_smem.subview(sched_idx * cutlass.Int32(8) + cutlass.Int32(1))).load())
    return t0, t1 & cutlass.Int32(0xFFFF), (t1 >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF)


@cute.jit
def _n_q_tiles(seqlen_q):
    return seqlen_q // cutlass.Int32(CFG.TILE_Q)


# ============================================================================
# Warp bodies
# ============================================================================


@cute.jit
def _softmax_warp_group(warp_idx, tmem_ptr_i32, bars, sched, sdS_raw, sStats_raw, sdV_raw, sdK_raw, seqlen_q, seqlen_kv, attn_scale, attn_scale_log2e) -> None:
    """8 compute warps (2 wg × 4); lane = kv row, wg owns a 64-q-col half.

    Per q-iter: P = exp2(scale·log2e·S − lse·log2e); fq(P) -> bf16 -> TMEM (dV A operand);
    dS = (scale·dP − scale·delta)·P -> bf16 -> sdS; after the dK/dQ MMAs commit, read the
    dQ partial out of the dP region in two 32-col halves and stage it as FP32 in the dS slot
    for the TMASTG reduce-add. Per kv-tile: dV/dK TMEM -> bf16 -> sdV/sdK.
    """
    nvvm.barrier_cta_sync(barrier_id=1, thread_count=32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))
    sdQ_raw = cutlass.Array(sdS_raw.data_ptr(), tileElems // 2, dtype=cutlass.Float32)

    _, head_idx, batch_idx = _boot_tile(sched)
    tid_in_wg = cute.arch.thread_idx()[0] & cutlass.Int32(127)
    wg_id = warp_idx // cutlass.Int32(CFG.SOFTMAX_WG_WARPS)
    q_half_off = wg_id * cutlass.Int32(_SMX_CHUNK)
    p_col_off = wg_id * cutlass.Int32(_SMX_CHUNK * CFG.BPE // 4)

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    s_full_state = PipelineState.start()
    dp_full_state = PipelineState.start()
    dsq_done_state = PipelineState.start()
    dq_stg_empty_state = PipelineState.start(phase=1)
    stats_full_state = PipelineState.start()
    dvk_ready_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        tmem_base = tmem_ptr_i32.load()
        n_q = _n_q_tiles(seqlen_q)

        for q_iter in cutlass.range(0, n_q, 1, unroll=1):
            stats_slot = stats_full_state.idx
            bars.mb_stats_full[stats_slot].wait(stats_full_state.phase)
            stats_base = stats_slot * cutlass.Int32(STATS_SLOT_ELEMS)

            # ---- 1) softmax: S -> P ; fq(P) -> bf16 -> TMEM ; p_ready ----
            bars.mb_s_acc_full.wait(s_full_state.phase)
            s_full_state = advance(s_full_state, 1)
            reg_S = tmem_load_tile(tmem_base + cutlass.Int32(LAYOUT.S_OFF) + q_half_off, num_elems=_SMX_CHUNK, ld_num=64)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            lse_vec = cutlass.Vector.from_elements(
                tuple(sStats_raw[stats_base + cutlass.Int32(STATS_LSE_OFF) + q_half_off + cutlass.Int32(i)] for i in range(_SMX_CHUNK)), cutlass.Float32
            )
            chunk_P = cute.math.exp2(reg_S.vec * attn_scale_log2e - lse_vec, fastmath=True)
            # Lanes are KV rows, registers are Q columns: NVFP4 groups are 16 adjacent lanes.
            chunk_P_dv = cutlass.Vector.from_elements(
                tuple(value for i in range(0, _SMX_CHUNK, 32) for value in _fake_quant_p_many(*(chunk_P[i + j] for j in range(32)))), cutlass.Float32
            )
            nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(tmem_base + cutlass.Int32(LAYOUT.P_OFF) + p_col_off, cutlass.Float32), chunk_P_dv.to(STORAGE_DTYPE))
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            bars.mb_p_ready.arrive()

            # ---- 2) dSoftmax: dP -> dS (FP32 P, scale folded) ----
            bars.mb_dp_full.wait(dp_full_state.phase)
            dp_full_state = advance(dp_full_state, 1)
            reg_dP = tmem_load_tile(tmem_base + cutlass.Int32(LAYOUT.dP_OFF) + q_half_off, num_elems=_SMX_CHUNK, ld_num=64)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            dot_vec = cutlass.Vector.from_elements(
                tuple(sStats_raw[stats_base + cutlass.Int32(STATS_DOT_OFF) + q_half_off + cutlass.Int32(i)] for i in range(_SMX_CHUNK)), cutlass.Float32
            )
            chunk_dS_f16 = ((reg_dP.vec * attn_scale - dot_vec) * chunk_P).to(STORAGE_DTYPE)
            bars.mb_stats_empty[stats_slot].arrive()
            stats_full_state = advance(stats_full_state, CFG.STATS_STAGES)

            # ---- 3) dS -> SMEM slot (A of dK and dQ).  The slot's previous content is
            #         the staged dQ half of iteration i-1: wait for its reduce to finish
            #         reading (same completion the half-0 staging below waits on). ----
            bars.mb_dq_stg_empty.wait(dq_stg_empty_state.phase)
            (sdS_raw.subview(wg_id * cutlass.Int32(SLAB_ELEMS) + tid_in_wg * cutlass.Int32(GRANU))).data_ptr().store_swizzled(
                chunk_dS_f16, alignment=128, swizzle=P_SMEM_SWIZZLE
            )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_ds_ready.arrive()

            # ---- 4) dQ partial drain: dP region -> FP32 -> dS slot (2 halves) ----
            bars.mb_dsq_done.wait(dsq_done_state.phase)
            dsq_done_state = advance(dsq_done_state, 1)
            for _h in cutlass.range_constexpr(DQ_HALVES):
                bars.mb_dq_stg_empty.wait(dq_stg_empty_state.phase)
                dq_stg_empty_state = advance(dq_stg_empty_state, 1)
                reg_dQ = tmem_load_tile(
                    tmem_base + cutlass.Int32(LAYOUT.dP_OFF) + wg_id * cutlass.Int32(_EPI_CHUNK) + cutlass.Int32(_h * DQ_STG_COLS),
                    num_elems=DQ_STG_COLS,
                    ld_num=DQ_STG_COLS,
                )
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                if cutlass.const_expr(_h == DQ_HALVES - 1):
                    bars.mb_dp_empty.arrive()  # dP region free for dP[i+1]
                (sdQ_raw.subview(wg_id * cutlass.Int32(DQ_SLAB_ELEMS) + tid_in_wg * cutlass.Int32(DQ_STG_COLS))).data_ptr().store_swizzled(
                    reg_dQ.vec, alignment=128, swizzle=P_SMEM_SWIZZLE
                )
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_dq_stg_full.arrive()

        # ---- dV / dK epilogue (per kv-tile) ----
        bars.mb_dvk_ready.wait(dvk_ready_state.phase)
        dvk_ready_state = advance(dvk_ready_state, 1)
        for _which in cutlass.range_constexpr(2):
            acc_off = LAYOUT.dV_OFF if _which == 0 else LAYOUT.dK_OFF
            reg_acc = tmem_load_tile(tmem_base + cutlass.Int32(acc_off) + wg_id * cutlass.Int32(_EPI_CHUNK), num_elems=_EPI_CHUNK, ld_num=64)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            acc_bf16 = reg_acc.vec.to(OUT_STORAGE_DTYPE)
            if cutlass.const_expr(_which == 0):
                (sdV_raw.subview(wg_id * cutlass.Int32(SLAB_ELEMS) + tid_in_wg * cutlass.Int32(GRANU))).data_ptr().store_swizzled(
                    acc_bf16, alignment=128, swizzle=P_SMEM_SWIZZLE
                )
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_dv_stg_full.arrive()
            else:
                (sdK_raw.subview(wg_id * cutlass.Int32(SLAB_ELEMS) + tid_in_wg * cutlass.Int32(GRANU))).data_ptr().store_swizzled(
                    acc_bf16, alignment=128, swizzle=P_SMEM_SWIZZLE
                )
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_dk_stg_full.arrive()
        bars.mb_acc_empty.arrive()

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_v = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        is_valid_tile = nxt_v & cutlass.Int32(1)
        _, head_idx, batch_idx = _decode_tile_payload(sched, sched_state.idx)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    bars.mb_tmem_dealloc.arrive()


@cute.jit
def _mma_warp(sQ, sQ_T, sdO, sdO_T, sK, sK_T, sV, sdS, sdS_T, tmem_ptr_i32, bars, sched, seqlen_q, seqlen_kv) -> None:
    """Per q-iter (all cta_group::1, M=128):  S = K·Qᵀ ; dP = V·dOᵀ ; dV += P·dO ;
    dK += dS·Q ; dQ_part = dSᵀ·K -> dP region.  In-order issue makes the S/P and
    dS reuse safe without extra barriers (see the 2-CTA kernel's notes)."""
    tmem_alloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)
    nvvm.barrier_cta_arrive(1, 32 * (CFG.SOFTMAX_WARPGROUPS * CFG.SOFTMAX_WG_WARPS + 1))

    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    q_full_state = PipelineState.start()
    do_full_state = PipelineState.start()
    k_full_state = PipelineState.start()
    v_full_state = PipelineState.start()
    p_ready_state = PipelineState.start()
    dp_empty_state = PipelineState.start(phase=1)
    ds_ready_state = PipelineState.start()
    acc_empty_state = PipelineState.start()

    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)
    tmem_S = tmem_raw.subview(cutlass.Int32(LAYOUT.S_OFF))
    tmem_P = tmem_raw.subview(cutlass.Int32(LAYOUT.P_OFF))
    tmem_dP = tmem_raw.subview(cutlass.Int32(LAYOUT.dP_OFF))
    tmem_dV = tmem_raw.subview(cutlass.Int32(LAYOUT.dV_OFF))
    tmem_dK = tmem_raw.subview(cutlass.Int32(LAYOUT.dK_OFF))

    def _desc(a_mn: bool, b_mn: bool):
        idesc = prims.Tcgen05InstrDesc.build(
            c_dtype=cutlass.Float32,
            a_dtype=STORAGE_DTYPE,
            b_dtype=STORAGE_DTYPE,
            n_dim=128,
            m_dim=128,
            a_major=1 if a_mn else 0,
            b_major=1 if b_mn else 0,
            k_dim=1,
        )
        return MmaDesc(
            M=128,
            N=128,
            K=128,
            bpe_a=CFG.BPE,
            bpe_b=CFG.BPE,
            tile_k_hw=CFG.TILE_K_HW,
            atranspose=a_mn,
            btranspose=b_mn,
            cta_group=1,
            idesc=idesc,
            kind=MMA_KIND,
        )

    desc_kk = _desc(False, False)  # S = K·Qᵀ, dP = V·dOᵀ (K = d)
    desc_kn = _desc(False, True)  # dV = P·dO, dK = dS·Q (K = q, B MN-major)
    desc_nn = _desc(True, True)  # dQ = dSᵀ·K (K = kv, both MN-major)

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        n_q = _n_q_tiles(seqlen_q)
        bars.mb_k_full.wait(k_full_state.phase)
        bars.mb_v_full.wait(v_full_state.phase)
        desc_K = sK[0].desc()
        desc_K_T = sK_T[0].desc()
        desc_V = sV[0].desc()

        for q_iter in cutlass.range(0, n_q, 1, unroll=1):
            qs = q_full_state.idx
            ds_ = do_full_state.idx
            # S = K·Q[i]ᵀ  (S region reuse is ordered by in-order issue after dV[i-1] read P)
            bars.mb_q_full[qs].wait(q_full_state.phase)
            mma_ss(desc_kk, desc_K, sQ[qs].desc(), tmem_S, accumulate=False)
            bars.mb_s_acc_full.arrive(cta_group=1, pred=nvvm.elect_sync())
            # dP = V·dO[i]ᵀ  (dP region: dQ_part[i-1] drained -> dp_empty)
            bars.mb_dp_empty.wait(dp_empty_state.phase)
            dp_empty_state = advance(dp_empty_state, 1)
            bars.mb_do_full[ds_].wait(do_full_state.phase)
            mma_ss(desc_kk, desc_V, sdO[ds_].desc(), tmem_dP, accumulate=False)
            bars.mb_dp_full.arrive(cta_group=1, pred=nvvm.elect_sync())
            # dV += fq(P)·dO[i]
            bars.mb_p_ready.wait(p_ready_state.phase)
            p_ready_state = advance(p_ready_state, 1)
            mma_ts(desc_kn, tmem_P, sdO_T[ds_].desc(), tmem_dV, accumulate=(q_iter > cutlass.Int32(0)))
            bars.mb_do_empty[ds_].arrive(cta_group=1, pred=nvvm.elect_sync())
            do_full_state = advance(do_full_state, CFG.STAGES_DO)
            # dK += dS·Q[i] ; dQ_part = dSᵀ·K
            bars.mb_ds_ready.wait(ds_ready_state.phase)
            ds_ready_state = advance(ds_ready_state, 1)
            mma_ss(desc_kn, sdS[0].desc(), sQ_T[qs].desc(), tmem_dK, accumulate=(q_iter > cutlass.Int32(0)))
            mma_ss(desc_nn, sdS_T[0].desc(), desc_K_T, tmem_dP, accumulate=False)
            elect_p = nvvm.elect_sync()
            bars.mb_q_empty[qs].arrive(cta_group=1, pred=elect_p)
            bars.mb_dsq_done.arrive(cta_group=1, pred=elect_p)
            q_full_state = advance(q_full_state, CFG.STAGES_Q)

        bars.mb_dvk_ready.arrive(cta_group=1, pred=nvvm.elect_sync())
        bars.mb_acc_empty.wait(acc_empty_state.phase)
        acc_empty_state = advance(acc_empty_state, 1)
        elect_p = nvvm.elect_sync()
        bars.mb_k_empty.arrive(cta_group=1, pred=elect_p)
        bars.mb_v_empty.arrive(cta_group=1, pred=elect_p)
        k_full_state = advance(k_full_state, 1)
        v_full_state = advance(v_full_state, 1)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_v = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        is_valid_tile = nxt_v & cutlass.Int32(1)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, LAYOUT.TOTAL_COLS, CTA_GROUP_KIND)


@cute.jit
def _tmastg_warp(
    tma_dv_desc,
    tma_dk_desc,
    tma_dq_desc,
    sdV,
    sdK,
    sdS_raw,
    sem_tensor,
    bars,
    sched,
    seqlen_q,
    seqlen_kv,
    n_qh,
    head_base,
    deterministic: cutlass.Constexpr[bool],
) -> None:
    """Per q-iter: two FP32 dQ halves -> cp.reduce.async.bulk.tensor.add into the dQ
    accumulator (deterministic: gated by the (b, h, q-tile) semaphore in kv-tile order).
    Per kv-tile: dV / dK TMA stores."""
    tma_dv = GmemTileTma(tma_dv_desc)
    tma_dk = GmemTileTma(tma_dk_desc)
    tma_dq_ptr = tma_dq_desc.get_ptr()
    sdQ_raw = cutlass.Array(sdS_raw.data_ptr(), tileElems // 2, dtype=cutlass.Float32)
    sem_base = Pointer(sem_tensor.iterator.raw_ptr(), dtype=cutlass.Int32)

    kv_tile, head_idx, batch_idx = _boot_tile(sched)
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    dq_full_state = PipelineState.start()
    dv_full_state = PipelineState.start()
    dk_full_state = PipelineState.start()

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        n_q = _n_q_tiles(seqlen_q)
        kv_row_base = kv_tile * cutlass.Int32(CFG.TILE_KV)
        full_head = cute.arch.make_warp_uniform(head_idx + head_base)
        sem_row = (batch_idx * n_qh + full_head) * n_q

        for q_iter in cutlass.range(0, n_q, 1, unroll=1):
            q_col_base = q_iter * cutlass.Int32(CFG.TILE_Q)
            if cutlass.const_expr(deterministic):
                # Release order: kv tiles 0, 1, 2, ... for this (b, h, q-tile).
                sem_ptr = sem_base + (sem_row + q_iter)
                if nvvm.elect_sync():
                    turn = cutlass.Int32(
                        nvvm.atomicrmw(nvvm.AtomicOp.ADD, sem_ptr, cutlass.Int32(0), mem_order=nvvm.MemOrder.ACQUIRE, syncscope=nvvm.MemScope.GPU)
                    )
                    while turn != kv_tile:
                        nvvm.nanosleep(cutlass.Int32(64))
                        turn = cutlass.Int32(
                            nvvm.atomicrmw(nvvm.AtomicOp.ADD, sem_ptr, cutlass.Int32(0), mem_order=nvvm.MemOrder.ACQUIRE, syncscope=nvvm.MemScope.GPU)
                        )
                nvvm.bar_warp_sync(cute.arch.FULL_MASK)
            for _h in cutlass.range_constexpr(DQ_HALVES):
                bars.mb_dq_stg_full.wait(dq_full_state.phase)
                dq_full_state = advance(dq_full_state, 1)
                if nvvm.elect_sync():
                    for _wg in cutlass.range_constexpr(CFG.SOFTMAX_WARPGROUPS):
                        d_coord = cutlass.Int32(_wg * _EPI_CHUNK + _h * DQ_STG_COLS)
                        nvvm.cp_async_bulk_tensor_reduce(
                            tma_dq_ptr,
                            sdQ_raw.subview(_wg * DQ_SLAB_ELEMS),
                            nvvm.TMARedux.ADD,
                            [d_coord, full_head, q_col_base, batch_idx],
                        )
                tma_store_commit()
                if cutlass.const_expr(deterministic and _h == DQ_HALVES - 1):
                    # Full completion (not just the SMEM read) before handing the turn on.
                    nvvm.cp_async_bulk_wait_group(0)
                    nvvm.bar_warp_sync(cute.arch.FULL_MASK)
                    if nvvm.elect_sync():
                        bars.mb_dq_stg_empty.arrive()
                        nvvm.atomicrmw(
                            nvvm.AtomicOp.EXCH,
                            sem_base + (sem_row + q_iter),
                            kv_tile + cutlass.Int32(1),
                            mem_order=nvvm.MemOrder.RELEASE,
                            syncscope=nvvm.MemScope.GPU,
                        )
                else:
                    tma_store_wait()
                    if nvvm.elect_sync():
                        bars.mb_dq_stg_empty.arrive()

        # dV / dK for this kv-tile (full-tensor head).
        bars.mb_dv_stg_full.wait(dv_full_state.phase)
        dv_full_state = advance(dv_full_state, 1)
        tma_store_tile(sdV[0], tma_dv(cutlass.Int32(0), full_head, kv_row_base, batch_idx))
        tma_store_commit()
        tma_store_wait()
        if nvvm.elect_sync():
            bars.mb_dv_stg_empty.arrive()
        bars.mb_dk_stg_full.wait(dk_full_state.phase)
        dk_full_state = advance(dk_full_state, 1)
        tma_store_tile(sdK[0], tma_dk(cutlass.Int32(0), full_head, kv_row_base, batch_idx))
        tma_store_commit()
        tma_store_wait()
        if nvvm.elect_sync():
            bars.mb_dk_stg_empty.arrive()

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_v = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        is_valid_tile = nxt_v & cutlass.Int32(1)
        kv_tile, head_idx, batch_idx = _decode_tile_payload(sched, sched_state.idx)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)


@cute.jit
def _tmaldg_warp(tma_q_desc, tma_do_desc, tma_k_desc, tma_v_desc, sQ, sdO, sK, sV, bars, sched, seqlen_q, seqlen_kv, qh_per_kh, head_base) -> None:
    tma_q = GmemTileTma(tma_q_desc)
    tma_do = GmemTileTma(tma_do_desc)
    tma_k = GmemTileTma(tma_k_desc)
    tma_v = GmemTileTma(tma_v_desc)

    kv_tile, head_idx, batch_idx = _boot_tile(sched)
    is_valid_tile = cutlass.Int32(1)
    sched_state = PipelineState.start()
    q_empty_state = PipelineState.start(phase=1)
    do_empty_state = PipelineState.start(phase=1)
    k_empty_state = PipelineState.start(phase=1)
    v_empty_state = PipelineState.start(phase=1)
    dv_storage_empty_state = PipelineState.start(phase=1)  # sdV aliases K
    dk_storage_empty_state = PipelineState.start(phase=1)  # sdK aliases V

    while is_valid_tile > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), CGA_SIZE)
        full_head = cute.arch.make_warp_uniform(head_idx + head_base)
        kv_head = cute.arch.make_warp_uniform(full_head // qh_per_kh)
        kv_row_base = kv_tile * cutlass.Int32(CFG.TILE_KV)
        n_q = _n_q_tiles(seqlen_q)

        # K, V — one-shot per kv-tile; their storage doubles as the previous tile's dV/dK staging.
        bars.mb_dv_stg_empty.wait(dv_storage_empty_state.phase)
        dv_storage_empty_state = advance(dv_storage_empty_state, 1)
        bars.mb_k_empty.wait(k_empty_state.phase)
        k_empty_state = advance(k_empty_state, 1)
        if nvvm.elect_sync():
            bars.mb_k_full.arrive(n_bytes=tileTmaBytes)
        tma_load_tile(sK[0], tma_k(cutlass.Int32(0), kv_head, kv_row_base, batch_idx), bars.mb_k_full.smem_ptr)
        bars.mb_dk_stg_empty.wait(dk_storage_empty_state.phase)
        dk_storage_empty_state = advance(dk_storage_empty_state, 1)
        bars.mb_v_empty.wait(v_empty_state.phase)
        v_empty_state = advance(v_empty_state, 1)
        if nvvm.elect_sync():
            bars.mb_v_full.arrive(n_bytes=tileTmaBytes)
        tma_load_tile(sV[0], tma_v(cutlass.Int32(0), kv_head, kv_row_base, batch_idx), bars.mb_v_full.smem_ptr)

        for q_iter in cutlass.range(0, n_q, 1, unroll=1):
            q_row_base = q_iter * cutlass.Int32(CFG.TILE_Q)
            qs = q_empty_state.idx
            bars.mb_q_empty[qs].wait(q_empty_state.phase)
            if nvvm.elect_sync():
                bars.mb_q_full[qs].arrive(n_bytes=tileTmaBytes)
            tma_load_tile(sQ[qs], tma_q(cutlass.Int32(0), full_head, q_row_base, batch_idx), bars.mb_q_full[qs].smem_ptr)
            q_empty_state = advance(q_empty_state, CFG.STAGES_Q)
            ds_ = do_empty_state.idx
            bars.mb_do_empty[ds_].wait(do_empty_state.phase)
            if nvvm.elect_sync():
                bars.mb_do_full[ds_].arrive(n_bytes=tileTmaBytes)
            tma_load_tile(sdO[ds_], tma_do(cutlass.Int32(0), full_head, q_row_base, batch_idx), bars.mb_do_full[ds_].smem_ptr)
            do_empty_state = advance(do_empty_state, CFG.STAGES_DO)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        nxt_v = (sched.tile_id_smem.subview(sched_state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        is_valid_tile = nxt_v & cutlass.Int32(1)
        kv_tile, head_idx, batch_idx = _decode_tile_payload(sched, sched_state.idx)
        sched_state = advance(sched_state, CFG.SCHEDULER_STAGES)


@cute.jit
def _scheduler_stats_warp(sched, bars, sStats_raw, lse_tensor, delta_tensor, attn_scale, seqlen_q, head_base) -> None:
    """try_cancel tile scheduler fused with the per-q-tile lse·log2e / delta·scale prefetch."""
    lse_arr = cutlass.make_array_view(lse_tensor)
    dot_arr = cutlass.make_array_view(delta_tensor)
    lane = cute.arch.thread_idx()[0] & cutlass.Int32(31)
    _PER_LANE = CFG.TILE_Q // 32

    _, cur_head, cur_batch = _boot_tile(sched)
    state = PipelineState.start()
    stats_empty_state = PipelineState.start(phase=1)
    is_valid = cutlass.Int32(1)

    while is_valid > cutlass.Int32(0):
        cur_full_head = cute.arch.make_warp_uniform(cur_head + head_base)
        n_q = _n_q_tiles(seqlen_q)
        for q_iter in cutlass.range(0, n_q, 1, unroll=1):
            q_col_base = q_iter * cutlass.Int32(CFG.TILE_Q)
            slot = stats_empty_state.idx
            bars.mb_stats_empty[slot].wait(stats_empty_state.phase)
            stats_empty_state = advance(stats_empty_state, CFG.STATS_STAGES)
            slot_base = slot * cutlass.Int32(STATS_SLOT_ELEMS)
            for j in cutlass.range_constexpr(_PER_LANE):
                col = lane + cutlass.Int32(j * 32)
                sStats_raw[slot_base + cutlass.Int32(STATS_LSE_OFF) + col] = lse_arr[cur_batch, cur_full_head, q_col_base + col] * cutlass.Float32(_LOG2E)
                sStats_raw[slot_base + cutlass.Int32(STATS_DOT_OFF) + col] = dot_arr[cur_batch, cur_full_head, q_col_base + col] * attn_scale
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_stats_full[slot].arrive()

        wait(sched.mb_read_tile_id.subview(state.idx), state.phase)
        if nvvm.elect_sync():
            arrive_expect_tx(sched.mb_scheduler.subview(state.idx), 16)
        if nvvm.elect_sync():
            nvvm.clusterlaunchcontrol_try_cancel(
                sched.tile_id_smem.subview(state.idx * cutlass.Int32(8)),
                sched.mb_scheduler.subview(state.idx),
                multicast=1,
            )
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(state.idx), state.phase)
        validity = (sched.tile_id_smem.subview(state.idx * cutlass.Int32(8) + cutlass.Int32(2))).load()
        is_valid = validity & cutlass.Int32(1)
        _, cur_head, cur_batch = _decode_tile_payload(sched, state.idx)
        state = advance(state, CFG.SCHEDULER_STAGES)


# ============================================================================
# Host
# ============================================================================


@cute.jit
def _host(
    q_tensor: cute.Tensor,  # [B, S_q, H, d] BF16 (BSHD)
    do_tensor: cute.Tensor,  # [B, S_q, H, d] BF16 view over caller BHSD
    k_tensor: cute.Tensor,  # [B, S_kv, H, d] BF16 (BSHD)
    v_tensor: cute.Tensor,  # [B, S_kv, H, d] BF16 (BSHD)
    dv_tensor: cute.Tensor,  # [B, S_kv, H, d] BF16 view over caller BHSD
    dk_tensor: cute.Tensor,  # [B, S_kv, H, d] BF16 view over caller BHSD
    dq_tensor: cute.Tensor,  # [B, S_q, H, d] FP32 view over BHSD accumulator (caller-zeroed)
    lse_tensor: cute.Tensor,  # [B, H, S_q] FP32
    delta_tensor: cute.Tensor,  # [B, H, S_q] FP32 raw
    sem_tensor: cute.Tensor,  # [B, H, S_q/128] Int32 (caller-zeroed; deterministic only)
    problem_size: Tuple[int, int, int, int, int, int],
    attn_scale: cutlass.Float32,
    attn_scale_log2e: cutlass.Float32,
    head_base: cutlass.Int32,
    deterministic: cutlass.Constexpr[bool],
    stream: _cuda_driver.CUstream,
) -> None:
    B, QH, KH, SQ, SKV, QH_CHUNK = problem_size
    order = (3, 2, 1, 0)
    swz = tmap.TensorMapSwizzle.s128b
    l2 = tmap.TensorMapL2Promotion.l2_128b
    box = (1, 128, 1, GRANU)  # 128 rows × 64 bf16 per slab
    dq_box = (1, 128, 1, DQ_STG_COLS)  # 128 q rows × 32 fp32
    tma_q_desc = tmap.create_tensor_map_tiled_from_view(q_tensor, box_dims=box, stride_order=order, swizzle=swz, l2_promotion=l2)
    tma_do_desc = tmap.create_tensor_map_tiled_from_view(do_tensor, box_dims=box, stride_order=order, swizzle=swz, l2_promotion=l2)
    tma_k_desc = tmap.create_tensor_map_tiled_from_view(k_tensor, box_dims=box, stride_order=order, swizzle=swz, l2_promotion=l2)
    tma_v_desc = tmap.create_tensor_map_tiled_from_view(v_tensor, box_dims=box, stride_order=order, swizzle=swz, l2_promotion=l2)
    tma_dv_desc = tmap.create_tensor_map_tiled_from_view(dv_tensor, box_dims=box, stride_order=order, swizzle=swz, l2_promotion=l2)
    tma_dk_desc = tmap.create_tensor_map_tiled_from_view(dk_tensor, box_dims=box, stride_order=order, swizzle=swz, l2_promotion=l2)
    tma_dq_desc = tmap.create_tensor_map_tiled_from_view(dq_tensor, box_dims=dq_box, stride_order=order, swizzle=swz, l2_promotion=l2)

    kv_tiles = (SKV + CFG.TILE_KV - 1) // CFG.TILE_KV
    _kernel(
        tma_q_desc,
        tma_do_desc,
        tma_k_desc,
        tma_v_desc,
        tma_dv_desc,
        tma_dk_desc,
        tma_dq_desc,
        lse_tensor,
        delta_tensor,
        sem_tensor,
        cutlass.Int32(SQ),
        cutlass.Int32(SKV),
        cutlass.Int32(QH),
        cutlass.Int32(B),
        cutlass.Int32(QH // KH),
        attn_scale,
        attn_scale_log2e,
        head_base,
        deterministic,
    ).launch(grid=(kv_tiles, QH_CHUNK, B), block=[CFG.THREADS_PER_CTA, 1, 1], cluster=(1, 1, 1), stream=stream)


def compile(b: int, qh: int, kh: int, sq: int, skv: int, qh_chunk: int = 0, deterministic: bool = True) -> Callable:
    """Compile the fused kernel for fixed plan-time shapes (B=1, MHA, S_q == S_kv, S % 128 == 0)."""
    if b != 1 or qh != kh or sq != skv or sq <= 0 or sq % CFG.TILE_KV:
        raise ValueError("SM100 QAT fused kernel requires B=1, MHA, and equal positive 128-aligned lengths")
    if qh_chunk == 0:
        qh_chunk = qh
    if qh_chunk <= 0 or qh <= 0 or qh % qh_chunk:
        raise ValueError("head chunk must be a positive divisor of H")
    d = CFG.D
    compact = lambda dt, shape, so: cute.runtime.make_fake_compact_tensor(dt, shape, stride_order=so, assumed_align=16)  # noqa: E731
    fake_q = compact(STORAGE_DTYPE, (b, sq, qh, d), (3, 2, 1, 0))
    fake_k = compact(STORAGE_DTYPE, (b, skv, kh, d), (3, 2, 1, 0))
    fake_v = compact(STORAGE_DTYPE, (b, skv, kh, d), (3, 2, 1, 0))
    bhsd = (sq * qh * d, d, sq * d, 1)
    fake_do = cute.runtime.make_fake_tensor(STORAGE_DTYPE, (b, sq, qh, d), bhsd, assumed_align=16)
    fake_dv = cute.runtime.make_fake_tensor(OUT_STORAGE_DTYPE, (b, skv, qh, d), bhsd, assumed_align=16)
    fake_dk = cute.runtime.make_fake_tensor(OUT_STORAGE_DTYPE, (b, skv, qh, d), bhsd, assumed_align=16)
    fake_dq = cute.runtime.make_fake_tensor(cutlass.Float32, (b, sq, qh, d), bhsd, assumed_align=16)
    fake_lse = compact(cutlass.Float32, (b, qh, sq), (2, 1, 0))
    fake_delta = compact(cutlass.Float32, (b, qh, sq), (2, 1, 0))
    fake_sem = compact(cutlass.Int32, (b, qh, sq // CFG.TILE_Q), (2, 1, 0))
    return cute.compile(
        _host,
        fake_q,
        fake_do,
        fake_k,
        fake_v,
        fake_dv,
        fake_dk,
        fake_dq,
        fake_lse,
        fake_delta,
        fake_sem,
        (b, qh, kh, sq, skv, qh_chunk),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Int32(0),
        deterministic,
        _cuda_driver.CUstream(0),
        options="--enable-tvm-ffi",
    )
