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
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout, AMDMFMALayout, warp_pipeline_stage
from triton.experimental.gluon.language.amd.rdna3 import wmma as wmma_rdna3
from triton.experimental.gluon.language.amd.rdna4 import wmma as wmma_rdna4
from triton.experimental.gluon.language.amd.cdna3 import mfma as mfma_cdna3
from triton.experimental.gluon.language.amd.cdna4 import mfma as mfma_cdna4
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async
from triton.experimental.gluon.language._layouts import DotOperandLayout, DistributedLinearLayout, PaddedSharedLayout


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
def issue_async_load_k(
    kt_smem, k_base, start_n,
    stride_kn, stride_kk,
    MASK_STEPS: gl.constexpr,
    MAX_SEQLENS_K: gl.constexpr,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr, ACTUAL_BLOCK_DMODEL: gl.constexpr,
    kt_async_layout: gl.constexpr,
):
    """Issue async load for K^T into shared memory."""
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)

    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kt_offs_n = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)
    kt_offsets = kt_offs_d[:, None] * stride_kk + (start_n + kt_offs_n[None, :]) * stride_kn

    if MASK_STEPS:
        kt_mask = (start_n + kt_offs_n[None, :]) < MAX_SEQLENS_K
        if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
            kt_mask = kt_mask & (kt_offs_d[:, None] < ACTUAL_BLOCK_DMODEL)
        cdna4_async.buffer_load_to_shared(kt_smem, k_base, kt_offsets, mask=kt_mask, other=0.0)
    else:
        cdna4_async.buffer_load_to_shared(kt_smem, k_base, kt_offsets)
    cdna4_async.commit_group()


@gluon.jit
def issue_async_load_v(
    v_smem, v_base, start_n,
    stride_vk, stride_vn,
    MASK_STEPS: gl.constexpr,
    MAX_SEQLENS_K: gl.constexpr,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr, ACTUAL_BLOCK_DMODEL: gl.constexpr,
    v_async_layout: gl.constexpr,
):
    """Issue async load for V into shared memory."""
    v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)

    v_offs_n = gl.arange(0, BLOCK_N, layout=v_offs_n_layout)
    v_offs_d = gl.arange(0, BLOCK_DMODEL, layout=v_offs_d_layout)
    v_offsets = (start_n + v_offs_n[:, None]) * stride_vk + v_offs_d[None, :] * stride_vn

    if MASK_STEPS:
        v_mask = (start_n + v_offs_n[:, None]) < MAX_SEQLENS_K
        if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
            v_mask = v_mask & (v_offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
        cdna4_async.buffer_load_to_shared(v_smem, v_base, v_offsets, mask=v_mask, other=0.0)
    else:
        cdna4_async.buffer_load_to_shared(v_smem, v_base, v_offsets)
    cdna4_async.commit_group()


@gluon.jit
def issue_async_load(
    kt_smem, v_smem, k_base, v_base, start_n,
    stride_kn, stride_kk, stride_vk, stride_vn,
    MASK_STEPS: gl.constexpr,
    MAX_SEQLENS_K: gl.constexpr,
    BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr, ACTUAL_BLOCK_DMODEL: gl.constexpr,
    kt_async_layout: gl.constexpr, v_async_layout: gl.constexpr,
):
    """Issue async loads for K^T and V into shared memory using buffer_load_to_shared."""
    issue_async_load_k(
        kt_smem, k_base, start_n,
        stride_kn, stride_kk,
        MASK_STEPS, MAX_SEQLENS_K,
        BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
        kt_async_layout,
    )
    issue_async_load_v(
        v_smem, v_base, start_n,
        stride_vk, stride_vn,
        MASK_STEPS, MAX_SEQLENS_K,
        BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
        v_async_layout,
    )


@gluon.jit
def compute_dot1_qk(
    q_dot, kt_smem,
    qk_scale: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    kt_dot_layout: gl.constexpr,
    mma_layout: gl.constexpr,
):
    """Dot1: Compute QK^T only. Returns UNSCALED qk scores.

    Scaling is deferred to compute_softmax to enable FMA fusion:
    qk * qk_scale - m_new can be compiled to a single FMA instruction.
    """
    # Load K^T from shared memory directly to DotOperandLayout.
    # PaddedSharedLayout ensures bank-conflict-free access.
    kt_dot = cdna4_async.load_shared_relaxed(kt_smem, kt_dot_layout)

    # Compute QK^T using MMA (Dot1).
    qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
    qk = do_mma("mfma_cdna4", q_dot, kt_dot, qk)

    # NOTE: Scaling deferred to compute_softmax for FMA fusion.
    return qk


@gluon.jit
def compute_softmax(
    acc, l_i, m_i, qk, start_n, start_m,
    qk_scale: gl.constexpr,
    MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    MASK_STEPS: gl.constexpr, IS_CAUSAL: gl.constexpr,
    mma_layout: gl.constexpr, mma_offs_n_col: gl.constexpr, mma_offs_m_row: gl.constexpr,
):
    """Online softmax: mask, compute max, exp, sum, scale accumulator.

    Takes UNSCALED qk and applies qk_scale during computation to enable FMA fusion.
    Pattern: qk * qk_scale - m_new compiles to FMA instruction.
    """
    if MASK_STEPS:
        # For masked steps, scale qk first, then apply masks.
        qk_scaled = qk * qk_scale

        # Apply causal mask.
        if IS_CAUSAL:
            causal_offs_n = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
            causal_offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
            causal_boundary = causal_offs_n[None, :] + MAX_SEQLENS_Q - MAX_SEQLENS_K
            causal_mask = causal_offs_m[:, None] >= causal_boundary
            qk_scaled = gl.where(causal_mask, qk_scaled, gl.full([BLOCK_M, BLOCK_N], float("-inf"),
                                                    dtype=gl.float32, layout=mma_layout))

        # Mask out-of-bounds K positions.
        bound_offs = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
        bound_mask = bound_offs[None, :] < MAX_SEQLENS_K
        qk_scaled = gl.where(bound_mask, qk_scaled, gl.full([BLOCK_M, BLOCK_N], float("-inf"),
                                                            dtype=gl.float32, layout=mma_layout))

        # Compute max of scaled qk.
        m_ij = gl.max(qk_scaled, axis=1)
        m_new = gl.maximum(m_i, m_ij)

        # Compute exp2(qk_scaled - m_new).
        p = gl.exp2(qk_scaled - m_new[:, None])
    else:
        # Unmasked path: use FMA pattern for qk * qk_scale - m_new.
        # Compute max of unscaled qk, then scale the result.
        m_ij = gl.max(qk, axis=1) * qk_scale
        m_new = gl.maximum(m_i, m_ij)

        # FMA: qk * qk_scale - m_new[:, None] in one operation.
        p = gl.exp2(qk * qk_scale - m_new[:, None])

    # Update running sum.
    l_ij = gl.sum(p, axis=1)
    alpha = gl.exp2(m_i - m_new)
    l_i = l_i * alpha + l_ij

    # Scale accumulator by alpha.
    acc = acc * alpha[:, None]

    # Update running max.
    m_i = m_new

    return acc, l_i, m_i, p


@gluon.jit
def compute_dot2_pv(
    acc, p, v_smem,
    p_dot_layout: gl.constexpr, v_dot_layout: gl.constexpr,
):
    """Dot2: Compute P @ V and accumulate."""
    # Load V from shared memory directly to DotOperandLayout.
    # PaddedSharedLayout ensures bank-conflict-free access.
    v_dot = cdna4_async.load_shared_relaxed(v_smem, v_dot_layout)

    # Accumulate P @ V using MMA (Dot2).
    p_cast = p.to(v_dot.dtype)
    p_dot = gl.convert_layout(p_cast, p_dot_layout)
    acc = do_mma("mfma_cdna4", p_dot, v_dot, acc)

    return acc


@gluon.jit
def compute_block(
    acc, l_i, m_i, q_dot, kt_smem, v_smem, start_n, start_m,
    qk_scale: gl.constexpr,
    MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    MASK_STEPS: gl.constexpr, IS_CAUSAL: gl.constexpr,
    kt_dot_layout: gl.constexpr, p_dot_layout: gl.constexpr, v_dot_layout: gl.constexpr,
    mma_layout: gl.constexpr, mma_offs_n_col: gl.constexpr, mma_offs_m_row: gl.constexpr,
):
    """Compute attention for one block using data from shared memory.
    Calls compute_dot1_qk, compute_softmax, then compute_dot2_pv.
    """
    qk = compute_dot1_qk(
        q_dot, kt_smem,
        qk_scale, BLOCK_M, BLOCK_N,
        kt_dot_layout, mma_layout,
    )
    acc, l_i, m_i, p = compute_softmax(
        acc, l_i, m_i, qk, start_n, start_m,
        qk_scale,
        MAX_SEQLENS_Q, MAX_SEQLENS_K,
        BLOCK_M, BLOCK_N, MASK_STEPS, IS_CAUSAL,
        mma_layout, mma_offs_n_col, mma_offs_m_row,
    )
    acc = compute_dot2_pv(acc, p, v_smem, p_dot_layout, v_dot_layout)
    return acc, l_i, m_i


@gluon.jit
def attn_fwd_inner_pipelined(
    acc, l_i, m_i, q_dot, k_base, v_base, start_m,
    stride_kn, stride_kk, stride_vk, stride_vn,
    block_start, block_end,
    kt_smem, v_smem,
    qk_scale: gl.constexpr,
    MAX_SEQLENS_Q: gl.constexpr, MAX_SEQLENS_K: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,
    MASK_STEPS: gl.constexpr, IS_CAUSAL: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    kt_async_layout: gl.constexpr, v_async_layout: gl.constexpr,
    kt_dot_layout: gl.constexpr, p_dot_layout: gl.constexpr, v_dot_layout: gl.constexpr,
    mma_layout: gl.constexpr, mma_offs_n_col: gl.constexpr, mma_offs_m_row: gl.constexpr,
):
    """
    Pipelined inner attention loop for CDNA4 with chained dot pattern.

    Structure for pingpong optimization (4 stages, matching gfx1250 FA pattern):
    - dot1 (prio=0): pure MMA QK^T, using kt_dot loaded in prev mem2
    - async_wait(V)
    - mem1 (prio=1): softmax + LDS load V + issue future K
    - dot2 (prio=0): pure MMA PV, using v_dot loaded in mem1
    - async_wait(K)
    - mem2 (prio=1): issue future V + LDS load K^T for next dot1

    kt_dot is loop-carried: loaded in mem2, consumed in next iteration's dot1.
    LDS reads are in memory stages so the pipeliner properly separates
    memory (high priority) from compute (low priority).
    """
    # Prologue: issue async loads for first NUM_STAGES blocks.
    # Issue K then V for each stage to maintain ordering: K0,V0,K1,V1,...
    # NOTE: No conditional here - caller must ensure block_end - block_start >= NUM_STAGES.
    # This is required for UpdateAsyncWaitCount pass to compute correct wait counts.
    for stage in gl.static_range(NUM_STAGES):
        start_n = (block_start + stage) * BLOCK_N
        issue_async_load_k(
            kt_smem.index(stage), k_base, start_n,
            stride_kn, stride_kk,
            MASK_STEPS, MAX_SEQLENS_K, BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
            kt_async_layout,
        )
        issue_async_load_v(
            v_smem.index(stage), v_base, start_n,
            stride_vk, stride_vn,
            MASK_STEPS, MAX_SEQLENS_K, BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
            v_async_layout,
        )

    # Wait counts for the 4-stage pattern.
    # After prologue: 2*NUM_STAGES loads in flight (K0,V0,K1,V1,...).
    # Prologue wait: wait(2*NUM_STAGES - 1) to get first K ready.
    # In the loop body, each wait is preceded by one new async issue,
    # so both wait_v and wait_k use the same count: 2*NUM_STAGES - 2.
    WAIT_INIT: gl.constexpr = 2 * NUM_STAGES - 1
    WAIT_LOOP: gl.constexpr = 2 * NUM_STAGES - 2

    # Prologue: wait for first K and load kt_dot for the first dot1.
    cdna4_async.wait_group(WAIT_INIT)
    kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)

    # Split into main loop + tail loop to eliminate control flow.
    # Main loop: always issues future loads (no conditional).
    # Tail loop: last NUM_STAGES iterations, no future loads to issue.
    main_loop_end = block_end - NUM_STAGES

    # Main loop: 4 stages with LDS reads absorbed into memory stages.
    # kt_dot is loop-carried (loaded in mem2, consumed in dot1).
    # NOTE: WarpPipeliner must run BEFORE loop unrolling (configured in compiler.py).
    for block_n in tl.range(block_start, main_loop_end, loop_unroll_factor=2):
        # Dot1: Pure MMA for QK^T (compute, low priority).
        # Uses kt_dot loaded in previous mem2 (or prologue for first iteration).
        with warp_pipeline_stage("dot1", priority=0):
            stage_idx = block_n % NUM_STAGES
            start_n = block_n * BLOCK_N
            future_start_n = (block_n + NUM_STAGES) * BLOCK_N
            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk = do_mma("mfma_cdna4", q_dot, kt_dot, qk)

        # Wait for V (between stages).
        cdna4_async.wait_group(WAIT_LOOP)

        # Mem1: Softmax + LDS load V + issue future K (memory, high priority).
        with warp_pipeline_stage("mem1", priority=1):
            v_dot = cdna4_async.load_shared_relaxed(v_smem.index(stage_idx), v_dot_layout)
            issue_async_load_k(
                kt_smem.index(stage_idx), k_base, future_start_n,
                stride_kn, stride_kk,
                MASK_STEPS, MAX_SEQLENS_K, BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
                kt_async_layout,
            )

        # Dot2: Pure MMA for PV (compute, low priority).
        # Uses v_dot loaded in mem1.
        with warp_pipeline_stage("dot2", priority=0):
            acc, l_i, m_i, p = compute_softmax(
                acc, l_i, m_i, qk, start_n, start_m,
                qk_scale,
                MAX_SEQLENS_Q, MAX_SEQLENS_K,
                BLOCK_M, BLOCK_N, MASK_STEPS, IS_CAUSAL,
                mma_layout, mma_offs_n_col, mma_offs_m_row,
            )
            p_cast = p.to(v_dot.dtype)
            p_dot = gl.convert_layout(p_cast, p_dot_layout)
            acc = do_mma("mfma_cdna4", p_dot, v_dot, acc)

        # Wait for K (between stages).
        cdna4_async.wait_group(WAIT_LOOP)

        # Mem2: Issue future V + LDS load K^T for next dot1 (memory, high priority).
        # Load from next iteration's buffer: mem1 already overwrote kt_smem[stage_idx].
        with warp_pipeline_stage("mem2", priority=1):
            issue_async_load_v(
                v_smem.index(stage_idx), v_base, future_start_n,
                stride_vk, stride_vn,
                MASK_STEPS, MAX_SEQLENS_K, BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
                v_async_layout,
            )
            next_stage_idx = (block_n + 1) % NUM_STAGES
            kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(next_stage_idx), kt_dot_layout)

    # Tail loop: last NUM_STAGES iterations, no future loads to issue.
    # Use gl.static_range to inline (unroll) tail iterations for better scheduling.
    # Each iteration uses descending wait counts instead of conservative wait_group(0).
    # At start of tail: 2*NUM_STAGES outstanding loads (K and V for each remaining block).
    # Iteration i: wait(2*(NUM_STAGES-i)-1) for K, wait(2*(NUM_STAGES-i)-2) for V.
    for tail_i in gl.static_range(NUM_STAGES):
        # Wait for K to be ready with descending wait count.
        # Outstanding loads decrease by 2 each iteration (consume K and V).
        cdna4_async.wait_group(2 * (NUM_STAGES - tail_i) - 1)

        # block_n = main_loop_end + tail_i = (block_end - NUM_STAGES) + tail_i
        # stage_idx = tail_i (since block_end is typically a multiple of NUM_STAGES).
        stage_idx = tail_i
        start_n = (main_loop_end + tail_i) * BLOCK_N

        # Compute Dot1 (QK^T).
        qk = compute_dot1_qk(
            q_dot, kt_smem.index(stage_idx),
            qk_scale, BLOCK_M, BLOCK_N,
            kt_dot_layout, mma_layout,
        )

        # Compute softmax.
        acc, l_i, m_i, p = compute_softmax(
            acc, l_i, m_i, qk, start_n, start_m,
            qk_scale,
            MAX_SEQLENS_Q, MAX_SEQLENS_K,
            BLOCK_M, BLOCK_N, MASK_STEPS, IS_CAUSAL,
            mma_layout, mma_offs_n_col, mma_offs_m_row,
        )

        # Wait for V to be ready with descending wait count.
        cdna4_async.wait_group(2 * (NUM_STAGES - tail_i) - 2)

        # Compute Dot2 (PV).
        acc = compute_dot2_pv(acc, p, v_smem.index(stage_idx), p_dot_layout, v_dot_layout)

    return acc, l_i, m_i


def get_gluon_cdna_autotune_configs():
    """Autotune configs for CDNA (MI series) GPUs."""
    return [
        # Pipelined config with NUM_STAGES=2 (CDNA4 only).
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'PRE_LOAD_V': False, 'NUM_STAGES': 4, 'waves_per_eu': 2}, num_warps=8),
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
        k_width: gl.constexpr = 32
        threads_per_warp: gl.constexpr = 64
    elif MMA_TYPE == "mfma_cdna4":
        mma_layout: gl.constexpr = AMDMFMALayout(version=4, instr_shape=[32, 32, 16],
                                                  transposed=True, warps_per_cta=[num_warps, 1])
        k_width: gl.constexpr = 32
        threads_per_warp: gl.constexpr = 64
    else:
        gl.static_assert(False, "Unknown MMA_TYPE")

    # Layouts for dot operands and blocked loads/stores.
    # Use smaller k_width for P@V dot to reduce permlanes in MMA→DotOperand conversion.
    # Triton uses kWidth=4 for P@V operands on CDNA4.
    # Q@K^T uses larger k_width for efficient memory loads.
    pv_k_width: gl.constexpr = 4 if MMA_TYPE == "mfma_cdna4" else k_width
    q_dot_layout: gl.constexpr = DotOperandLayout(operand_index=0, parent=mma_layout, k_width=k_width)
    kt_dot_layout: gl.constexpr = DotOperandLayout(operand_index=1, parent=mma_layout, k_width=k_width)
    p_dot_layout: gl.constexpr = DotOperandLayout(operand_index=0, parent=mma_layout, k_width=pv_k_width)
    v_dot_layout: gl.constexpr = DotOperandLayout(operand_index=1, parent=mma_layout, k_width=pv_k_width)

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

    # Load Q tile [BLOCK_M, BLOCK_DMODEL].
    # We load through shared memory to avoid LDS shuffle when converting to dot_op layout.
    # This matches Triton's approach: load to blocked -> store to swizzled shared -> load to dot_op.
    q_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=16, order=[1, 0])
    q_smem = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_M, BLOCK_DMODEL], layout=q_smem_layout)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = offs_m[:, None] < MAX_SEQLENS_Q
    if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
        q_mask = q_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    q = gl.load(q_ptrs, mask=q_mask, other=0.0)

    # Store Q to shared memory and load directly to dot operand layout (no LDS shuffle needed).
    q_smem.store(q)
    q_dot = q_smem.load(q_dot_layout)

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

    # Use pipelined path for CDNA4 with NUM_STAGES > 1 and BLOCK_DMODEL >= 128.
    # The BLOCK_DMODEL constraint exists because direct-to-LDS async copy requires
    # coalesced writes where all 64 lane bits map to the fast dimension.
    # With vec=2 (32-bit writes) and 64 threads: 2*64=128 consecutive elements per warp.
    # The fast dimension (BLOCK_DMODEL) must be >= 128 to satisfy this constraint.
    USE_PIPELINED: gl.constexpr = (MMA_TYPE == "mfma_cdna4") and (NUM_STAGES > 1) and (BLOCK_DMODEL >= 128)

    if USE_PIPELINED:
        # Async copy layout configuration using PaddedSharedLayout for bank conflict-free access.
        # This matches Triton's FA layout which uses row permutation (high bits before low bits
        # for the non-fast dimension) to spread accesses across different banks.
        #
        # Memory layout:
        # - K^T [BLOCK_DMODEL, BLOCK_N] = [128, 64]: dim0 fast, order=[0, 1]
        # - V [BLOCK_N, BLOCK_DMODEL] = [64, 128]: dim1 fast, order=[1, 0]
        #
        # Triton uses #ttg.padded_shared with [512:+8] for K^T and [512:+32] for V.
        # The offset_bases follow a row permutation pattern for bank conflict avoidance.

        # K^T offset_bases: 7 bases for dim0 (128), then 6 reordered bases for dim1 (64).
        # Reordering: high bits [0,16], [0,32] come before low bits [0,1], [0,2], [0,4], [0,8].
        kt_offset_bases: gl.constexpr = [
            [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0],  # dim0 (7 bases)
            [0, 16], [0, 32],  # dim1 high bits
            [0, 1], [0, 2], [0, 4], [0, 8]  # dim1 low bits
        ]
        kt_async_smem_layout: gl.constexpr = PaddedSharedLayout(
            interval_padding_pairs=[[512, 8]],
            offset_bases=kt_offset_bases,
            cga_layout=[],
            shape=[BLOCK_DMODEL, BLOCK_N])

        # V offset_bases: 7 bases for dim1 (128), then 6 reordered bases for dim0 (64).
        v_offset_bases: gl.constexpr = [
            [0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64],  # dim1 (7 bases)
            [16, 0], [32, 0],  # dim0 high bits
            [1, 0], [2, 0], [4, 0], [8, 0]  # dim0 low bits
        ]
        v_async_smem_layout: gl.constexpr = PaddedSharedLayout(
            interval_padding_pairs=[[512, 32]],
            offset_bases=v_offset_bases,
            cga_layout=[],
            shape=[BLOCK_N, BLOCK_DMODEL])

        # DistributedLinearLayout for async copy offsets.
        # Following CoalesceAsyncCopy algorithm: distribute offset_bases to reg/lane/warp.
        # For vec=8 (3 reg bases), threads_per_warp=64 (6 lane bases), num_warps=8 (3 warp bases).
        # Remaining bases go to additional registers.
        #
        # K^T: 13 bases total -> reg(3+1), lane(6), warp(3)
        kt_async_layout: gl.constexpr = DistributedLinearLayout(
            reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8]],
            lane_bases=[[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]],
            warp_bases=[[0, 1], [0, 2], [0, 4]],
            block_bases=[],
            shape=[BLOCK_DMODEL, BLOCK_N])

        # V: 13 bases total -> reg(3+1), lane(6), warp(3)
        v_async_layout: gl.constexpr = DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
            lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]],
            warp_bases=[[1, 0], [2, 0], [4, 0]],
            block_bases=[],
            shape=[BLOCK_N, BLOCK_DMODEL])

        # Allocate multi-buffered shared memory for pipelining.
        # Use a single 3D buffer with NUM_STAGES as the first dimension.
        # Access individual stages with .index(stage) to avoid type conversion issues.
        kt_smem = gl.allocate_shared_memory(
            Q.dtype.element_ty, [NUM_STAGES, BLOCK_DMODEL, BLOCK_N], layout=kt_async_smem_layout)
        v_smem = gl.allocate_shared_memory(
            Q.dtype.element_ty, [NUM_STAGES, BLOCK_N, BLOCK_DMODEL], layout=v_async_smem_layout)

        # Process full blocks (no masking needed - faster).
        # Pipelined prologue requires at least NUM_STAGES blocks.
        if n_full_blocks >= NUM_STAGES:
            acc, l_i, m_i = attn_fwd_inner_pipelined(
                acc, l_i, m_i, q_dot, k_base, v_base, start_m,
                stride_kn, stride_kk, stride_vk, stride_vn,
                0, n_full_blocks,
                kt_smem, v_smem,
                qk_scale,
                MAX_SEQLENS_Q, MAX_SEQLENS_K,
                BLOCK_M, BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
                False, False,  # MASK_STEPS=False, IS_CAUSAL=False for full blocks.
                NUM_STAGES,
                kt_async_layout, v_async_layout,
                kt_dot_layout, p_dot_layout, v_dot_layout,
                mma_layout, mma_offs_n_col, mma_offs_m_row,
            )

        # Process masked blocks (need causal and/or boundary masking).
        # If n_full_blocks < NUM_STAGES, include those in masked path (start from 0).
        masked_start = n_full_blocks if n_full_blocks >= NUM_STAGES else 0
        remaining_blocks = n_blocks - masked_start
        if remaining_blocks >= NUM_STAGES:
            acc, l_i, m_i = attn_fwd_inner_pipelined(
                acc, l_i, m_i, q_dot, k_base, v_base, start_m,
                stride_kn, stride_kk, stride_vk, stride_vn,
                masked_start, n_blocks,
                kt_smem, v_smem,
                qk_scale,
                MAX_SEQLENS_Q, MAX_SEQLENS_K,
                BLOCK_M, BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
                True, IS_CAUSAL,  # MASK_STEPS=True for masked blocks.
                NUM_STAGES,
                kt_async_layout, v_async_layout,
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
    # Compute reciprocal first to avoid 256*128 element-wise divisions.
    # This reduces divisions from O(BLOCK_M * BLOCK_DMODEL) to O(BLOCK_M).
    l_recip = 1.0 / l_i
    acc = acc * l_recip[:, None]

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
