"""
Flash Attention v2 - Gluon Implementation (AMD RDNA3)
=====================================================

This is a simplified implementation of Flash Attention using Triton's Gluon language
with AMD RDNA3 WMMA instructions for matrix multiplication.

Supports: basic forward pass with optional causal masking.
Does NOT support: VARLEN, INT8, dropout, ALiBi, bias, persistent mode.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout
from triton.experimental.gluon.language.amd.rdna3 import wmma
from triton.experimental.gluon.language._layouts import DotOperandLayout


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
                   num_warps: gl.constexpr):
    """
    Gluon Flash Attention Forward Kernel with AMD RDNA3 WMMA.

    Grid: (num_heads_q, num_m_blocks, batch)
    """
    # Validate unsupported features at compile time.
    gl.static_assert(not VARLEN, "VARLEN not supported in Gluon implementation")
    gl.static_assert(not INT8, "INT8 not supported in Gluon implementation")
    gl.static_assert(not ENABLE_DROPOUT, "Dropout not supported in Gluon implementation")
    gl.static_assert(not USE_ALIBI, "ALiBi not supported in Gluon implementation")
    gl.static_assert(not USE_BIAS, "Bias not supported in Gluon implementation")
    gl.static_assert(not PERSISTENT, "Persistent mode not supported in Gluon implementation")

    # Program IDs.
    off_h_q = gl.program_id(0)
    start_m = gl.program_id(1)
    off_z = gl.program_id(2)

    # MQA/GQA head mapping.
    off_h_k = off_h_q * HK // HQ

    # AMD WMMA layout for RDNA3 (version=1).
    # WMMA instruction shape is [16, 16, 16] by default.
    qk_wmma_layout: gl.constexpr = AMDWMMALayout(
        version=1,
        transposed=False,
        warps_per_cta=[num_warps, 1],
        instr_shape=[16, 16, 16],
    )

    acc_wmma_layout: gl.constexpr = AMDWMMALayout(
        version=1,
        transposed=False,
        warps_per_cta=[num_warps, 1],
        instr_shape=[16, 16, 16],
    )

    # DotOperandLayouts for WMMA.
    # For QK^T: Q [M, D] @ K^T [D, N] -> QK [M, N].
    q_dot_layout: gl.constexpr = DotOperandLayout(operand_index=0, parent=qk_wmma_layout, k_width=16)
    kt_dot_layout: gl.constexpr = DotOperandLayout(operand_index=1, parent=qk_wmma_layout, k_width=16)

    # For P @ V: P [M, N] @ V [N, D] -> O [M, D].
    p_dot_layout: gl.constexpr = DotOperandLayout(operand_index=0, parent=acc_wmma_layout, k_width=16)
    v_dot_layout: gl.constexpr = DotOperandLayout(operand_index=1, parent=acc_wmma_layout, k_width=16)

    # Blocked layout for loads/stores.
    blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[32, 1],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )

    # 1D slice layouts.
    offs_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked_layout)
    offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=blocked_layout)
    offs_n_row_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked_layout)
    offs_n_col_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=blocked_layout)

    # Offset ranges.
    offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=offs_m_layout)
    offs_d = gl.arange(0, BLOCK_DMODEL, layout=offs_d_layout)

    # Base pointers.
    q_base = Q + off_z * stride_qz + off_h_q * stride_qh
    k_base = K + off_z * stride_kz + off_h_k * stride_kh
    v_base = V + off_z * stride_vz + off_h_k * stride_vh

    # Load Q tile [BLOCK_M, BLOCK_DMODEL].
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = offs_m[:, None] < MAX_SEQLENS_Q
    if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
        q_mask = q_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    q = gl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize accumulators with WMMA-compatible slice layout.
    wmma_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=qk_wmma_layout)
    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=wmma_m_layout)
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=wmma_m_layout)
    acc = gl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=gl.float32, layout=acc_wmma_layout)

    # Scale factor (log2(e) * sm_scale for exp2 trick).
    qk_scale: gl.constexpr = SM_SCALE * 1.44269504089

    # Number of K/V blocks to process.
    n_blocks: gl.constexpr = (MAX_SEQLENS_K + BLOCK_N - 1) // BLOCK_N

    # WMMA-compatible slice layouts for masking (defined outside loop to avoid redefinition).
    wmma_offs_n_col: gl.constexpr = gl.SliceLayout(dim=0, parent=qk_wmma_layout)
    wmma_offs_m_row: gl.constexpr = gl.SliceLayout(dim=1, parent=qk_wmma_layout)

    # Main loop over K/V blocks.
    # Note: Gluon doesn't support `continue` in static_range, so we rely on
    # the causal mask to mask out blocks that would have been skipped.
    for block_n in gl.static_range(n_blocks):
        start_n = block_n * BLOCK_N
        offs_n = start_n + gl.arange(0, BLOCK_N, layout=offs_n_row_layout)

        # Load K tile [BLOCK_N, BLOCK_DMODEL].
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_mask = offs_n[:, None] < MAX_SEQLENS_K
        if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
            k_mask = k_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
        k = gl.load(k_ptrs, mask=k_mask, other=0.0)

        # Compute QK^T using WMMA.
        # Q is [BLOCK_M, BLOCK_DMODEL], K is [BLOCK_N, BLOCK_DMODEL].
        # Transpose K: [BLOCK_N, BLOCK_DMODEL] -> [BLOCK_DMODEL, BLOCK_N].
        k_t = gl.permute(k, [1, 0])

        # Convert to dot operand layouts for WMMA.
        q_dot = gl.convert_layout(q, q_dot_layout)
        kt_dot = gl.convert_layout(k_t, kt_dot_layout)
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=qk_wmma_layout)
        qk = wmma(q_dot, kt_dot, qk)

        # Scale QK scores.
        qk = qk * qk_scale

        # Apply causal mask.
        if IS_CAUSAL:
            causal_offs_n = start_n + gl.arange(0, BLOCK_N, layout=wmma_offs_n_col)
            causal_offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=wmma_offs_m_row)
            causal_boundary = causal_offs_n[None, :] + MAX_SEQLENS_Q - MAX_SEQLENS_K
            causal_mask = causal_offs_m[:, None] >= causal_boundary
            neg_inf = gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=qk_wmma_layout)
            qk = gl.where(causal_mask, qk, neg_inf)

        # Mask out-of-bounds K positions.
        bound_offs = start_n + gl.arange(0, BLOCK_N, layout=wmma_offs_n_col)
        bound_mask = bound_offs[None, :] < MAX_SEQLENS_K
        neg_inf2 = gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=qk_wmma_layout)
        qk = gl.where(bound_mask, qk, neg_inf2)

        # Online softmax: compute new running max.
        m_ij = gl.max(qk, axis=1)
        m_new = gl.where(m_ij > m_i, m_ij, m_i)

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

        # Load V tile [BLOCK_N, BLOCK_DMODEL].
        v_ptrs = v_base + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
        v = gl.load(v_ptrs, mask=k_mask, other=0.0)

        # Accumulate P @ V using WMMA.
        # P is [BLOCK_M, BLOCK_N] (fp32), V is [BLOCK_N, BLOCK_DMODEL] (fp16).
        # Convert P to fp16 for WMMA.
        p_f16 = p.to(gl.float16)
        p_dot = gl.convert_layout(p_f16, p_dot_layout)
        v_dot = gl.convert_layout(v, v_dot_layout)
        acc = wmma(p_dot, v_dot, acc)

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
