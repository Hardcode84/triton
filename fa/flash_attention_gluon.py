"""
Flash Attention v2 - Gluon Implementation (AMD)
===============================================

This is a simplified implementation of Flash Attention using Triton's Gluon language
with configurable AMD matrix core instructions (WMMA/MFMA).

Supports: basic forward pass with optional causal masking.
Does NOT support: VARLEN, INT8, dropout, ALiBi, bias, persistent mode.

MMA_TYPE options:
- "wmma_rdna3": AMD RDNA3 WMMA (gfx1100, gfx1101)
- "wmma_rdna4": AMD RDNA4 WMMA (gfx1200, gfx1201)
- "mfma_cdna3": AMD CDNA3 MFMA (gfx942)
- "mfma_cdna4": AMD CDNA4 MFMA (gfx950)

Optimizations applied:
- Q, K, V loaded through shared memory (global->shared->registers)
- Q layout conversion hoisted out of loop
- K loaded with transposed layout directly (no permute)
- PRE_LOAD_V: V loaded early for pipelining
- Pointer arithmetic instead of offset recomputation
- gl.maximum for running max computation
- Full/masked block split to skip masking for most blocks
- Autotuning for BLOCK_M, BLOCK_N, num_warps, PRE_LOAD_V
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout, AMDMFMALayout
from triton.experimental.gluon.language.amd.rdna3 import wmma as wmma_rdna3
from triton.experimental.gluon.language.amd.rdna4 import wmma as wmma_rdna4
from triton.experimental.gluon.language.amd.cdna3 import mfma as mfma_cdna3
from triton.experimental.gluon.language.amd.cdna4 import mfma as mfma_cdna4
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async
from triton.experimental.gluon.language._layouts import DotOperandLayout


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in (
        'gfx950', 'gfx940', 'gfx941', 'gfx942', 'gfx90a', 'gfx908')


def is_rdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in (
        "gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201")


def get_mma_type_for_arch(arch: str) -> str:
    """Get the appropriate MMA type for the given GPU architecture."""
    if arch.startswith("gfx110"):
        return "wmma_rdna3"
    elif arch.startswith("gfx120"):
        return "wmma_rdna4"
    elif arch in ("gfx940", "gfx941", "gfx942"):
        return "mfma_cdna3"
    elif arch == "gfx950":
        return "mfma_cdna4"
    else:
        raise ValueError(f"Unsupported GPU architecture: {arch}")


@gluon.jit
def do_mma(MMA_TYPE: gl.constexpr, a, b, c):
    """Dispatch to the appropriate MMA function based on MMA_TYPE."""
    if MMA_TYPE == "wmma_rdna3":
        return wmma_rdna3(a, b, c)
    elif MMA_TYPE == "wmma_rdna4":
        return wmma_rdna4(a, b, c)
    elif MMA_TYPE == "mfma_cdna3":
        return mfma_cdna3(a, b, c)
    elif MMA_TYPE == "mfma_cdna4":
        return mfma_cdna4(a, b, c)


@gluon.jit
def attn_fwd_inner(
    acc, l_i, m_i, q_dot, kt_ptrs, v_ptrs, offs_n, offs_d,
    kt_offs_d, kt_offs_n, start_m,
    stride_kn, stride_vk,
    block_start, block_end,
    kt_smem, v_smem,
    qk_scale: gl.constexpr,
    MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,
    PRE_LOAD_V: gl.constexpr, MASK_STEPS: gl.constexpr, IS_CAUSAL: gl.constexpr,
    MMA_TYPE: gl.constexpr,
    kt_blocked_layout: gl.constexpr, blocked_layout: gl.constexpr,
    kt_dot_layout: gl.constexpr, p_dot_layout: gl.constexpr, v_dot_layout: gl.constexpr,
    mma_layout: gl.constexpr, mma_offs_n_col: gl.constexpr, mma_offs_m_row: gl.constexpr,
):
    """Inner attention loop over K/V blocks with shared memory staging."""
    for block_n in range(block_start, block_end):
        start_n = block_n * BLOCK_N

        # PRE_LOAD_V: Load V early to allow pipelining.
        # Global load -> shared store -> shared load.
        if PRE_LOAD_V:
            if MASK_STEPS:
                v_mask = (start_n + offs_n[:, None]) < MAX_SEQLENS_K
                if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
                    v_mask = v_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
                v_global = gl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v_global = gl.load(v_ptrs)
            v_smem.store(v_global)

        # Load K^T: global -> shared -> registers.
        if MASK_STEPS:
            kt_mask = (start_n + kt_offs_n[None, :]) < MAX_SEQLENS_K
            if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
                kt_mask = kt_mask & (kt_offs_d[:, None] < ACTUAL_BLOCK_DMODEL)
            kt_global = gl.load(kt_ptrs, mask=kt_mask, other=0.0)
        else:
            kt_global = gl.load(kt_ptrs)
        kt_smem.store(kt_global)

        # Load K^T from shared memory.
        k_t = kt_smem.load(kt_blocked_layout)

        # Compute QK^T using MMA.
        kt_dot = gl.convert_layout(k_t, kt_dot_layout)
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(MMA_TYPE, q_dot, kt_dot, qk)

        # Scale QK scores.
        qk = qk * qk_scale

        # Apply causal mask (only for masked blocks).
        if MASK_STEPS and IS_CAUSAL:
            causal_offs_n = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
            causal_offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
            causal_boundary = causal_offs_n[None, :] + MAX_SEQLENS_Q - MAX_SEQLENS_K
            causal_mask = causal_offs_m[:, None] >= causal_boundary
            qk = gl.where(causal_mask, qk, gl.full([BLOCK_M, BLOCK_N], float("-inf"),
                                                    dtype=gl.float32, layout=mma_layout))

        # Mask out-of-bounds K positions (only for masked blocks).
        if MASK_STEPS:
            bound_offs = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
            bound_mask = bound_offs[None, :] < MAX_SEQLENS_K
            qk = gl.where(bound_mask, qk, gl.full([BLOCK_M, BLOCK_N], float("-inf"),
                                                   dtype=gl.float32, layout=mma_layout))

        # Online softmax: compute new running max.
        m_ij = gl.max(qk, axis=1)
        m_new = gl.maximum(m_i, m_ij)

        # Compute exp2(qk - m_new) for numerical stability.
        p = gl.exp2(qk - m_new[:, None])

        # Update running sum.
        l_ij = gl.sum(p, axis=1)
        alpha = gl.exp2(m_i - m_new)
        l_i = l_i * alpha + l_ij

        # Scale accumulator by alpha.
        acc = acc * alpha[:, None]

        # Update running max.
        m_i = m_new

        # Load V tile if not pre-loaded: global -> shared -> registers.
        if not PRE_LOAD_V:
            if MASK_STEPS:
                v_mask = (start_n + offs_n[:, None]) < MAX_SEQLENS_K
                if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
                    v_mask = v_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
                v_global = gl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v_global = gl.load(v_ptrs)
            v_smem.store(v_global)

        # Load V from shared memory.
        v = v_smem.load(blocked_layout)

        # Accumulate P @ V using MMA.
        p_cast = p.to(v.dtype)
        p_dot = gl.convert_layout(p_cast, p_dot_layout)
        v_dot = gl.convert_layout(v, v_dot_layout)
        acc = do_mma(MMA_TYPE, p_dot, v_dot, acc)

        # Advance pointers.
        kt_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk

    return acc, l_i, m_i, kt_ptrs, v_ptrs


@gluon.jit
def issue_async_load(
    kt_smem, v_smem, k_base, v_base, start_n,
    stride_kn, stride_kk, stride_vk, stride_vn,
    MASK_STEPS: gl.constexpr,
    MAX_SEQLENS_K: gl.constexpr,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr, ACTUAL_BLOCK_DMODEL: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Issue async loads for K^T and V into shared memory using buffer_load_to_shared.

    Uses layouts compatible with async copy (size_per_thread * bits = 128).
    For fp16, size_per_thread=8 gives 128 bits.
    """
    # K^T layout [BLOCK_DMODEL, BLOCK_N]: size_per_thread=[1, 8] for 128 bits with fp16.
    kt_async_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[8, 8],
        warps_per_cta=[num_warps, 1], order=[1, 0])
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)

    # V layout [BLOCK_N, BLOCK_DMODEL]: size_per_thread=[1, 8] for 128 bits with fp16.
    v_async_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[8, 8],
        warps_per_cta=[num_warps, 1], order=[1, 0])
    v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)

    # Construct K^T offsets [BLOCK_DMODEL, BLOCK_N] (element offsets from base).
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kt_offs_n = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)
    kt_offsets = kt_offs_d[:, None] * stride_kk + (start_n + kt_offs_n[None, :]) * stride_kn

    # Construct V offsets [BLOCK_N, BLOCK_DMODEL].
    v_offs_n = gl.arange(0, BLOCK_N, layout=v_offs_n_layout)
    v_offs_d = gl.arange(0, BLOCK_DMODEL, layout=v_offs_d_layout)
    v_offsets = (start_n + v_offs_n[:, None]) * stride_vk + v_offs_d[None, :] * stride_vn

    if MASK_STEPS:
        kt_mask = (start_n + kt_offs_n[None, :]) < MAX_SEQLENS_K
        if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
            kt_mask = kt_mask & (kt_offs_d[:, None] < ACTUAL_BLOCK_DMODEL)
        cdna4_async.buffer_load_to_shared(kt_smem, k_base, kt_offsets, mask=kt_mask, other=0.0)
        v_mask = (start_n + v_offs_n[:, None]) < MAX_SEQLENS_K
        if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
            v_mask = v_mask & (v_offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
        cdna4_async.buffer_load_to_shared(v_smem, v_base, v_offsets, mask=v_mask, other=0.0)
    else:
        cdna4_async.buffer_load_to_shared(kt_smem, k_base, kt_offsets)
        cdna4_async.buffer_load_to_shared(v_smem, v_base, v_offsets)


@gluon.jit
def compute_block(
    acc, l_i, m_i, q_dot, kt_smem, v_smem, start_n, start_m,
    qk_scale: gl.constexpr,
    MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    MASK_STEPS: gl.constexpr, IS_CAUSAL: gl.constexpr,
    kt_blocked_layout: gl.constexpr, blocked_layout: gl.constexpr,
    kt_dot_layout: gl.constexpr, p_dot_layout: gl.constexpr, v_dot_layout: gl.constexpr,
    mma_layout: gl.constexpr, mma_offs_n_col: gl.constexpr, mma_offs_m_row: gl.constexpr,
):
    """Compute attention for one block using data from shared memory."""
    # Load K^T from shared memory.
    k_t = cdna4_async.load_shared_relaxed(kt_smem, kt_blocked_layout)

    # Compute QK^T using MMA.
    kt_dot = gl.convert_layout(k_t, kt_dot_layout)
    qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
    qk = do_mma("mfma_cdna4", q_dot, kt_dot, qk)

    # Scale QK scores.
    qk = qk * qk_scale

    # Apply causal mask (only for masked blocks).
    if MASK_STEPS and IS_CAUSAL:
        causal_offs_n = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
        causal_offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
        causal_boundary = causal_offs_n[None, :] + MAX_SEQLENS_Q - MAX_SEQLENS_K
        causal_mask = causal_offs_m[:, None] >= causal_boundary
        qk = gl.where(causal_mask, qk, gl.full([BLOCK_M, BLOCK_N], float("-inf"),
                                                dtype=gl.float32, layout=mma_layout))

    # Mask out-of-bounds K positions (only for masked blocks).
    if MASK_STEPS:
        bound_offs = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
        bound_mask = bound_offs[None, :] < MAX_SEQLENS_K
        qk = gl.where(bound_mask, qk, gl.full([BLOCK_M, BLOCK_N], float("-inf"),
                                               dtype=gl.float32, layout=mma_layout))

    # Online softmax: compute new running max.
    m_ij = gl.max(qk, axis=1)
    m_new = gl.maximum(m_i, m_ij)

    # Compute exp2(qk - m_new) for numerical stability.
    p = gl.exp2(qk - m_new[:, None])

    # Update running sum.
    l_ij = gl.sum(p, axis=1)
    alpha = gl.exp2(m_i - m_new)
    l_i = l_i * alpha + l_ij

    # Scale accumulator by alpha.
    acc = acc * alpha[:, None]

    # Update running max.
    m_i = m_new

    # Load V from shared memory.
    v = cdna4_async.load_shared_relaxed(v_smem, blocked_layout)

    # Accumulate P @ V using MMA.
    p_cast = p.to(v.dtype)
    p_dot = gl.convert_layout(p_cast, p_dot_layout)
    v_dot = gl.convert_layout(v, v_dot_layout)
    acc = do_mma("mfma_cdna4", p_dot, v_dot, acc)

    return acc, l_i, m_i


@gluon.jit
def attn_fwd_inner_pipelined(
    acc, l_i, m_i, q_dot, k_base, v_base, start_m,
    stride_kn, stride_kk, stride_vk, stride_vn,
    block_start, block_end,
    kt_smem_stages, v_smem_stages,
    qk_scale: gl.constexpr,
    MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,
    MASK_STEPS: gl.constexpr, IS_CAUSAL: gl.constexpr,
    NUM_STAGES: gl.constexpr, num_warps: gl.constexpr,
    kt_blocked_layout: gl.constexpr, blocked_layout: gl.constexpr,
    kt_dot_layout: gl.constexpr, p_dot_layout: gl.constexpr, v_dot_layout: gl.constexpr,
    mma_layout: gl.constexpr, mma_offs_n_col: gl.constexpr, mma_offs_m_row: gl.constexpr,
):
    """
    Pipelined inner attention loop for CDNA4 using async memory operations.
    NUM_STAGES controls the pipeline depth (number of buffers).
    """
    # Prologue: issue async loads for first NUM_STAGES blocks.
    for stage in gl.static_range(NUM_STAGES):
        block_n = block_start + stage
        if block_n < block_end:
            start_n = block_n * BLOCK_N
            issue_async_load(
                kt_smem_stages[stage], v_smem_stages[stage],
                k_base, v_base, start_n,
                stride_kn, stride_kk, stride_vk, stride_vn,
                MASK_STEPS, MAX_SEQLENS_K, BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
                num_warps,
            )

    # Main loop: process blocks with full pipeline.
    # Wait count is constant: (NUM_STAGES - 1) * 2 loads remain in flight.
    WAIT_STAGES: gl.constexpr = (NUM_STAGES - 1) * 2
    for block_n in range(block_start, block_end):
        # stage = block_n % NUM_STAGES
        kt_smem_stage = kt_smem_stages[0]
        v_smem_stage = v_smem_stages[0]
        for s in gl.static_range(NUM_STAGES):
            if block_n % NUM_STAGES == s:
                kt_smem_stage = kt_smem_stages[s]
                v_smem_stage = v_smem_stages[s]
        start_n = block_n * BLOCK_N

        # Wait for this stage's loads to complete.
        cdna4_async.async_wait(WAIT_STAGES)

        # Compute attention for this block.
        acc, l_i, m_i = compute_block(
            acc, l_i, m_i, q_dot,
            kt_smem_stage, v_smem_stage,
            start_n, start_m,
            qk_scale, MAX_SEQLENS_Q, MAX_SEQLENS_K,
            BLOCK_M, BLOCK_N, MASK_STEPS, IS_CAUSAL,
            kt_blocked_layout, blocked_layout,
            kt_dot_layout, p_dot_layout, v_dot_layout,
            mma_layout, mma_offs_n_col, mma_offs_m_row,
        )

        # Issue async load for future block (block_n + NUM_STAGES).
        future_block = block_n + NUM_STAGES
        if future_block < block_end:
            future_start_n = future_block * BLOCK_N
            issue_async_load(
                kt_smem_stage, v_smem_stage,
                k_base, v_base, future_start_n,
                stride_kn, stride_kk, stride_vk, stride_vn,
                MASK_STEPS, MAX_SEQLENS_K, BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
                num_warps,
            )

    # Final wait to ensure all loads are done.
    cdna4_async.async_wait(0)

    return acc, l_i, m_i


def get_gluon_cdna_autotune_configs():
    """Autotune configs for CDNA (MI series) GPUs."""
    return [
        # Pipelined configs with NUM_STAGES > 1 (CDNA4 only).
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 2, 'waves_per_eu': 2}, num_warps=8),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 2}, num_warps=4),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 3}, num_warps=4),
        # Non-pipelined configs (NUM_STAGES=1).
        # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 1, 'waves_per_eu': 2}, num_warps=8),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'PRE_LOAD_V': True, 'NUM_STAGES': 1}, num_warps=4),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 1}, num_warps=4),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'PRE_LOAD_V': True, 'NUM_STAGES': 1}, num_warps=4),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 1}, num_warps=4),
    ]


def get_gluon_rdna_autotune_configs():
    """Autotune configs for RDNA (RX series) GPUs."""
    return [
        # RDNA uses non-pipelined path (NUM_STAGES=1).
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 1}, num_warps=4),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'PRE_LOAD_V': True, 'NUM_STAGES': 1}, num_warps=2),
        # triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'PRE_LOAD_V': True, 'NUM_STAGES': 1}, num_warps=2),
        # triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'PRE_LOAD_V': False, 'NUM_STAGES': 1}, num_warps=2),
    ]


def get_gluon_autotune_configs():
    """Get autotune configs based on current GPU architecture."""
    if is_rdna():
        return get_gluon_rdna_autotune_configs()
    elif is_cdna():
        return get_gluon_cdna_autotune_configs()
    else:
        # Fallback configs.
        return [triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'PRE_LOAD_V': False, 'NUM_STAGES': 1}, num_warps=4)]


# Autotune keys: parameters that affect which config is best.
GLUON_AUTOTUNE_KEYS = ['IS_CAUSAL', 'MAX_SEQLENS_Q', 'MAX_SEQLENS_K', 'ACTUAL_BLOCK_DMODEL', 'HQ', 'HK']


@triton.autotune(
    configs=get_gluon_autotune_configs(),
    key=GLUON_AUTOTUNE_KEYS,
)
@gluon.jit
def gluon_attn_fwd(Q, K, V, bias, SM_SCALE: gl.constexpr, L, Out,
                   stride_qz, stride_qh, stride_qm, stride_qk,
                   stride_kz, stride_kh, stride_kn, stride_kk,
                   stride_vz, stride_vh, stride_vk, stride_vn,
                   stride_oz, stride_oh, stride_om, stride_on,
                   stride_bz, stride_bh, stride_bm, stride_bn,
                   stride_az, stride_ah,
                   Q_descale, K_descale, P_scale, P_descale, V_descale,
                   cu_seqlens_q, cu_seqlens_k,
                   dropout_p, philox_seed,
                   PERSISTENT: gl.constexpr, PERSISTENT_DYNAMIC: gl.constexpr,
                   atomic_counter,
                   NUM_CU: gl.constexpr, GRID_CU_MULTIP: gl.constexpr, B: gl.constexpr,
                   philox_offset_base, encoded_softmax, alibi_slopes,
                   HQ: gl.constexpr, HK: gl.constexpr,
                   ACTUAL_BLOCK_DMODEL: gl.constexpr,
                   MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
                   VARLEN: gl.constexpr, IS_CAUSAL: gl.constexpr,
                   BLOCK_M: gl.constexpr, BLOCK_DMODEL: gl.constexpr, BLOCK_N: gl.constexpr,
                   PRE_LOAD_V: gl.constexpr, USE_BIAS: gl.constexpr,
                   ENABLE_DROPOUT: gl.constexpr, RETURN_ENCODED_SOFTMAX: gl.constexpr,
                   USE_ALIBI: gl.constexpr, INT8: gl.constexpr,
                   USE_P_SCALE: gl.constexpr, INT8_KV: gl.constexpr,
                   MMA_TYPE: gl.constexpr, NUM_STAGES: gl.constexpr):
    """
    Gluon Flash Attention Forward Kernel with configurable AMD MMA.
    Grid: (num_heads_q, num_m_blocks, batch)

    Note: num_warps is set via autotune config, accessed via gl.num_warps().
    """
    # Get num_warps from runtime (set by autotune).
    num_warps: gl.constexpr = gl.num_warps()
    # Validate unsupported features at compile time.
    gl.static_assert(not VARLEN, "VARLEN not supported in Gluon implementation")
    gl.static_assert(not INT8, "INT8 not supported in Gluon implementation")
    gl.static_assert(not ENABLE_DROPOUT, "Dropout not supported in Gluon implementation")
    gl.static_assert(not USE_ALIBI, "ALiBi not supported in Gluon implementation")
    gl.static_assert(not USE_BIAS, "Bias not supported in Gluon implementation")
    gl.static_assert(not PERSISTENT, "Persistent mode not supported in Gluon implementation")

    gl.assume(stride_qz >= 0)
    gl.assume(stride_qh >= 0)
    gl.assume(stride_qm >= 0)
    gl.assume(stride_qk >= 0)
    gl.assume(stride_kz >= 0)
    gl.assume(stride_kh >= 0)
    gl.assume(stride_kn >= 0)
    gl.assume(stride_kk >= 0)
    gl.assume(stride_bz >= 0)
    gl.assume(stride_bh >= 0)
    gl.assume(stride_bm >= 0)
    gl.assume(stride_bn >= 0)
    gl.assume(stride_vz >= 0)
    gl.assume(stride_vh >= 0)
    gl.assume(stride_vk >= 0)
    gl.assume(stride_vn >= 0)
    gl.assume(stride_oz >= 0)
    gl.assume(stride_oh >= 0)
    gl.assume(stride_om >= 0)
    gl.assume(stride_on >= 0)

    # Program IDs.
    off_h_q = gl.program_id(0)
    start_m = gl.program_id(1)
    off_z = gl.program_id(2)

    # MQA/GQA head mapping.
    off_h_k = off_h_q * HK // HQ

    # Configure MMA layout based on MMA_TYPE.
    if MMA_TYPE == "wmma_rdna3":
        mma_layout: gl.constexpr = AMDWMMALayout(version=1, transposed=True,
                                                  warps_per_cta=[num_warps, 1], instr_shape=[16, 16, 16])
        k_width: gl.constexpr = 16
        threads_per_warp: gl.constexpr = 32
    elif MMA_TYPE == "wmma_rdna4":
        mma_layout: gl.constexpr = AMDWMMALayout(version=2, transposed=True,
                                                  warps_per_cta=[num_warps, 1], instr_shape=[16, 16, 16])
        k_width: gl.constexpr = 16
        threads_per_warp: gl.constexpr = 32
    elif MMA_TYPE == "mfma_cdna3":
        mma_layout: gl.constexpr = AMDMFMALayout(version=3, instr_shape=[16, 16, 16],
                                                  transposed=True, warps_per_cta=[num_warps, 1])
        k_width: gl.constexpr = 4
        threads_per_warp: gl.constexpr = 64
    elif MMA_TYPE == "mfma_cdna4":
        mma_layout: gl.constexpr = AMDMFMALayout(version=4, instr_shape=[32, 32, 16],
                                                  transposed=True, warps_per_cta=[num_warps, 1])
        k_width: gl.constexpr = 4
        threads_per_warp: gl.constexpr = 64
    else:
        gl.static_assert(False, "Unknown MMA_TYPE")

    # Layouts for dot operands and blocked loads/stores.
    q_dot_layout: gl.constexpr = DotOperandLayout(operand_index=0, parent=mma_layout, k_width=k_width)
    kt_dot_layout: gl.constexpr = DotOperandLayout(operand_index=1, parent=mma_layout, k_width=k_width)
    p_dot_layout: gl.constexpr = DotOperandLayout(operand_index=0, parent=mma_layout, k_width=k_width)
    v_dot_layout: gl.constexpr = DotOperandLayout(operand_index=1, parent=mma_layout, k_width=k_width)

    blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[threads_per_warp // 4, 4],
        warps_per_cta=[num_warps, 1], order=[1, 0])
    kt_blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1], threads_per_warp=[1, threads_per_warp],
        warps_per_cta=[1, num_warps], order=[0, 1])

    # Slice layouts.
    offs_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked_layout)
    offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=blocked_layout)
    offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked_layout)
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_blocked_layout)
    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_blocked_layout)
    mma_offs_n_col: gl.constexpr = gl.SliceLayout(dim=0, parent=mma_layout)
    mma_offs_m_row: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)
    mma_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_layout)

    # Offset ranges.
    offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=offs_m_layout)
    offs_d = gl.arange(0, BLOCK_DMODEL, layout=offs_d_layout)
    offs_n = gl.arange(0, BLOCK_N, layout=offs_n_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kt_offs_n = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)

    # Base pointers.
    q_base = Q + off_z * stride_qz + off_h_q * stride_qh
    k_base = K + off_z * stride_kz + off_h_k * stride_kh
    v_base = V + off_z * stride_vz + off_h_k * stride_vh

    # Load Q tile [BLOCK_M, BLOCK_DMODEL] directly (only loaded once, no need for shared memory).
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = offs_m[:, None] < MAX_SEQLENS_Q
    if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
        q_mask = q_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    q = gl.load(q_ptrs, mask=q_mask, other=0.0)

    # Convert Q to dot operand layout once (hoisted from loop).
    q_dot = gl.convert_layout(q, q_dot_layout)

    # Initialize accumulators.
    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout)
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_m_layout)
    acc = gl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=gl.float32, layout=mma_layout)

    # Scale factor (log2(e) * sm_scale for exp2 trick).
    qk_scale: gl.constexpr = SM_SCALE * 1.44269504089

    # Number of K/V blocks to process.
    # For causal attention, we can skip blocks where all positions are masked.
    n_blocks_total: gl.constexpr = (MAX_SEQLENS_K + BLOCK_N - 1) // BLOCK_N
    n_extra_tokens: gl.constexpr = MAX_SEQLENS_K % BLOCK_N
    padded_block_k: gl.constexpr = n_extra_tokens != 0
    is_modulo_mn: gl.constexpr = not padded_block_k and (MAX_SEQLENS_Q % BLOCK_M == 0)

    if IS_CAUSAL:
        # Causal boundary: positions where m >= n + (seqlen_q - seqlen_k).
        causal_block_limit = (start_m + 1) * BLOCK_M + MAX_SEQLENS_K - MAX_SEQLENS_Q
        n_blocks = gl.minimum(n_blocks_total, (causal_block_limit + BLOCK_N - 1) // BLOCK_N)
        # There are always at least BLOCK_M // BLOCK_N masked blocks.
        # Additionally there might be one more due to dissimilar seqlens.
        masked_blocks: gl.constexpr = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        n_blocks = n_blocks_total
        # Padding on Q does not need to be masked in the FA loop.
        masked_blocks: gl.constexpr = 1 if padded_block_k else 0

    # Clamp masked_blocks to n_blocks (may exceed for small sequences).
    masked_blocks_clamped = gl.minimum(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks_clamped

    # Initialize K^T and V pointers (K loaded transposed: [D, N]).
    kt_ptrs = k_base + kt_offs_d[:, None] * stride_kk + kt_offs_n[None, :] * stride_kn
    v_ptrs = v_base + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn

    # Use pipelined path for CDNA4 with NUM_STAGES > 1.
    USE_PIPELINED: gl.constexpr = (MMA_TYPE == "mfma_cdna4") and (NUM_STAGES > 1)

    if USE_PIPELINED:
        # Async copy requires simple swizzling (within warp boundary).
        # Use vec=1, per_phase=1, max_phase=1 for async copy compatibility.
        kt_async_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0, 1])
        v_async_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])

        # Allocate multi-buffered shared memory for pipelining.
        # Explicit allocation for each supported NUM_STAGES value.
        if NUM_STAGES == 2:
            kt_smem_0 = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_DMODEL, BLOCK_N], layout=kt_async_smem_layout)
            kt_smem_1 = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_DMODEL, BLOCK_N], layout=kt_async_smem_layout)
            v_smem_0 = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_N, BLOCK_DMODEL], layout=v_async_smem_layout)
            v_smem_1 = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_N, BLOCK_DMODEL], layout=v_async_smem_layout)
            kt_smem_stages = (kt_smem_0, kt_smem_1)
            v_smem_stages = (v_smem_0, v_smem_1)
        elif NUM_STAGES == 3:
            kt_smem_0 = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_DMODEL, BLOCK_N], layout=kt_async_smem_layout)
            kt_smem_1 = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_DMODEL, BLOCK_N], layout=kt_async_smem_layout)
            kt_smem_2 = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_DMODEL, BLOCK_N], layout=kt_async_smem_layout)
            v_smem_0 = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_N, BLOCK_DMODEL], layout=v_async_smem_layout)
            v_smem_1 = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_N, BLOCK_DMODEL], layout=v_async_smem_layout)
            v_smem_2 = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_N, BLOCK_DMODEL], layout=v_async_smem_layout)
            kt_smem_stages = (kt_smem_0, kt_smem_1, kt_smem_2)
            v_smem_stages = (v_smem_0, v_smem_1, v_smem_2)
        elif NUM_STAGES == 4:
            kt_smem_0 = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_DMODEL, BLOCK_N], layout=kt_async_smem_layout)
            kt_smem_1 = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_DMODEL, BLOCK_N], layout=kt_async_smem_layout)
            kt_smem_2 = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_DMODEL, BLOCK_N], layout=kt_async_smem_layout)
            kt_smem_3 = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_DMODEL, BLOCK_N], layout=kt_async_smem_layout)
            v_smem_0 = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_N, BLOCK_DMODEL], layout=v_async_smem_layout)
            v_smem_1 = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_N, BLOCK_DMODEL], layout=v_async_smem_layout)
            v_smem_2 = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_N, BLOCK_DMODEL], layout=v_async_smem_layout)
            v_smem_3 = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_N, BLOCK_DMODEL], layout=v_async_smem_layout)
            kt_smem_stages = (kt_smem_0, kt_smem_1, kt_smem_2, kt_smem_3)
            v_smem_stages = (v_smem_0, v_smem_1, v_smem_2, v_smem_3)
        else:
            gl.static_assert(False, "NUM_STAGES must be 2, 3, or 4 for pipelined path")

        # Process full blocks (no masking needed - faster).
        if n_full_blocks > 0:
            acc, l_i, m_i = attn_fwd_inner_pipelined(
                acc, l_i, m_i, q_dot, k_base, v_base, start_m,
                stride_kn, stride_kk, stride_vk, stride_vn,
                0, n_full_blocks,
                kt_smem_stages, v_smem_stages,
                qk_scale,
                MAX_SEQLENS_Q, MAX_SEQLENS_K,
                BLOCK_M, BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
                False, False,  # MASK_STEPS=False, IS_CAUSAL=False for full blocks.
                NUM_STAGES, num_warps,
                kt_blocked_layout, blocked_layout,
                kt_dot_layout, p_dot_layout, v_dot_layout,
                mma_layout, mma_offs_n_col, mma_offs_m_row,
            )

        # Process masked blocks (need causal and/or boundary masking).
        if masked_blocks > 0:
            acc, l_i, m_i = attn_fwd_inner_pipelined(
                acc, l_i, m_i, q_dot, k_base, v_base, start_m,
                stride_kn, stride_kk, stride_vk, stride_vn,
                n_full_blocks, n_blocks,
                kt_smem_stages, v_smem_stages,
                qk_scale,
                MAX_SEQLENS_Q, MAX_SEQLENS_K,
                BLOCK_M, BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
                True, IS_CAUSAL,  # MASK_STEPS=True for masked blocks.
                NUM_STAGES, num_warps,
                kt_blocked_layout, blocked_layout,
                kt_dot_layout, p_dot_layout, v_dot_layout,
                mma_layout, mma_offs_n_col, mma_offs_m_row,
            )
    else:
        # Non-pipelined path: single-buffered shared memory with optimized swizzling.
        kt_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=16, order=[0, 1])
        v_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=16, order=[1, 0])
        kt_smem = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_DMODEL, BLOCK_N], layout=kt_smem_layout)
        v_smem = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_N, BLOCK_DMODEL], layout=v_smem_layout)

        # Process full blocks (no masking needed - faster).
        if n_full_blocks > 0:
            acc, l_i, m_i, kt_ptrs, v_ptrs = attn_fwd_inner(
                acc, l_i, m_i, q_dot, kt_ptrs, v_ptrs, offs_n, offs_d,
                kt_offs_d, kt_offs_n, start_m,
                stride_kn, stride_vk,
                0, n_full_blocks,
                kt_smem, v_smem,
                qk_scale,
                MAX_SEQLENS_Q, MAX_SEQLENS_K,
                BLOCK_M, BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
                PRE_LOAD_V, False, False,  # MASK_STEPS=False, IS_CAUSAL=False for full blocks.
                MMA_TYPE,
                kt_blocked_layout, blocked_layout,
                kt_dot_layout, p_dot_layout, v_dot_layout,
                mma_layout, mma_offs_n_col, mma_offs_m_row,
            )

        # Process masked blocks (need causal and/or boundary masking).
        if masked_blocks > 0:
            acc, l_i, m_i, kt_ptrs, v_ptrs = attn_fwd_inner(
                acc, l_i, m_i, q_dot, kt_ptrs, v_ptrs, offs_n, offs_d,
                kt_offs_d, kt_offs_n, start_m,
                stride_kn, stride_vk,
                n_full_blocks, n_blocks,
                kt_smem, v_smem,
                qk_scale,
                MAX_SEQLENS_Q, MAX_SEQLENS_K,
                BLOCK_M, BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
                PRE_LOAD_V, True, IS_CAUSAL,  # MASK_STEPS=True for masked blocks.
                MMA_TYPE,
                kt_blocked_layout, blocked_layout,
                kt_dot_layout, p_dot_layout, v_dot_layout,
                mma_layout, mma_offs_n_col, mma_offs_m_row,
            )

    # Normalize by softmax sum.
    acc = acc / l_i[:, None]

    # Store output.
    o_base = Out + off_z * stride_oz + off_h_q * stride_oh
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    o_mask = offs_m[:, None] < MAX_SEQLENS_Q
    if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
        o_mask = o_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    # Convert layout for store.
    acc_blocked = gl.convert_layout(acc, blocked_layout)
    gl.store(o_ptrs, acc_blocked.to(Out.dtype.element_ty), mask=o_mask)

    # Store log-sum-exp for backward pass.
    l_ptrs = L + off_z * HQ * MAX_SEQLENS_Q + off_h_q * MAX_SEQLENS_Q + offs_m
    l_mask = offs_m < MAX_SEQLENS_Q
    # Convert from log2 scale back to natural log.
    lse = m_i / 1.44269504089 + gl.log2(l_i) / 1.44269504089
    # Convert from WMMA slice layout to blocked slice layout for store.
    lse_blocked = gl.convert_layout(lse, offs_m_layout)
    gl.store(l_ptrs, lse_blocked, mask=l_mask)
