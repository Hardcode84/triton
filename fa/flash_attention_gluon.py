"""
Flash Attention v2 - Gluon Implementation
=========================================

This is a simplified implementation of Flash Attention using Triton's Gluon language.
Supports: basic forward pass with optional causal masking.
Does NOT support: VARLEN, INT8, dropout, ALiBi, bias, persistent mode.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def gluon_attn_fwd_simple(
    Q, K, V, Out, L,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    HQ: gl.constexpr, HK: gl.constexpr,
    N_CTX_Q: gl.constexpr, N_CTX_K: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_DMODEL: gl.constexpr,
    SM_SCALE: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    num_warps: gl.constexpr,
):
    """
    Simple Gluon Flash Attention Forward Kernel.

    Grid: (num_m_blocks, batch, num_heads_q)
    """
    # Program IDs.
    start_m = gl.program_id(0)
    off_z = gl.program_id(1)
    off_h_q = gl.program_id(2)

    # MQA/GQA head mapping.
    off_h_k = off_h_q * HK // HQ

    # Create 2D blocked layout for tiles.
    acc_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, BLOCK_DMODEL // 32],
        threads_per_warp=[32, 1],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )

    # 1D layouts for offsets.
    offs_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=acc_layout)
    offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=acc_layout)

    # Create offset ranges.
    offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=offs_m_layout)
    offs_d = gl.arange(0, BLOCK_DMODEL, layout=offs_d_layout)

    # Base pointers.
    q_base = Q + off_z * stride_qz + off_h_q * stride_qh
    k_base = K + off_z * stride_kz + off_h_k * stride_kh
    v_base = V + off_z * stride_vz + off_h_k * stride_vh

    # Load Q tile [BLOCK_M, BLOCK_DMODEL].
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = offs_m[:, None] < N_CTX_Q
    q = gl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize accumulators.
    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=offs_m_layout)
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=offs_m_layout)
    acc = gl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=gl.float32, layout=acc_layout)

    # Scale for numerical stability (log2(e) * sm_scale).
    qk_scale: gl.constexpr = SM_SCALE * 1.44269504089

    # Number of K/V blocks.
    n_blocks = (N_CTX_K + BLOCK_N - 1) // BLOCK_N
    if IS_CAUSAL:
        # Only process up to causal boundary.
        n_blocks_causal = ((start_m + 1) * BLOCK_M + N_CTX_K - N_CTX_Q + BLOCK_N - 1) // BLOCK_N
        n_blocks = min(n_blocks, n_blocks_causal)

    # KV block layout.
    kv_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, BLOCK_DMODEL // 32],
        threads_per_warp=[32, 1],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kv_layout)

    # QK layout for attention scores.
    qk_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[32, 1],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )

    # Main loop over K/V blocks.
    for block_n in range(0, n_blocks):
        start_n = block_n * BLOCK_N
        offs_n = start_n + gl.arange(0, BLOCK_N, layout=offs_n_layout)

        # Load K tile [BLOCK_N, BLOCK_DMODEL].
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_mask = offs_n[:, None] < N_CTX_K
        k = gl.load(k_ptrs, mask=k_mask, other=0.0)

        # Compute QK^T = Q @ K^T -> [BLOCK_M, BLOCK_N].
        # Manual computation since we need specific layouts for dot_fma.
        # For simplicity, compute element-wise: qk[i,j] = sum_d(q[i,d] * k[j,d]).
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=qk_layout)

        # Naive dot product implementation.
        # In a full implementation, this would use dot_fma with proper layouts.
        for d in range(BLOCK_DMODEL):
            q_d = q[:, d:d+1]  # [BLOCK_M, 1]
            k_d = k[:, d:d+1]  # [BLOCK_N, 1]
            # Broadcast and multiply.
            q_broadcast = gl.convert_layout(q_d, gl.SliceLayout(dim=1, parent=qk_layout))
            k_broadcast = gl.convert_layout(k_d, gl.SliceLayout(dim=0, parent=qk_layout))
            qk = qk + q_broadcast[:, None] * k_broadcast[None, :]

        # Scale QK.
        qk = qk * qk_scale

        # Apply causal mask.
        if IS_CAUSAL:
            causal_offs_n = start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=qk_layout))
            causal_offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(dim=1, parent=qk_layout))
            causal_boundary = causal_offs_n[None, :] + N_CTX_Q - N_CTX_K
            causal_mask = causal_offs_m[:, None] >= causal_boundary
            qk = gl.where(causal_mask, qk, gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=qk_layout))

        # Mask out-of-bounds K positions.
        n_mask_offs = start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=qk_layout))
        n_mask = n_mask_offs[None, :] < N_CTX_K
        qk = gl.where(n_mask, qk, gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=qk_layout))

        # Online softmax: compute new max.
        m_ij = gl.reduce(qk, axis=1, fn="max")
        m_ij = gl.where(m_ij > m_i, m_ij, m_i)

        # Compute exp(qk - m_ij) using exp2.
        qk_shifted = qk - m_ij[:, None]
        p = gl.exp2(qk_shifted)

        # Update running sum.
        l_ij = gl.reduce(p, axis=1, fn="sum")
        alpha = gl.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        # Scale accumulator by alpha.
        acc = acc * alpha[:, None]

        # Load V tile [BLOCK_N, BLOCK_DMODEL].
        v_ptrs = v_base + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
        v = gl.load(v_ptrs, mask=k_mask, other=0.0)

        # Accumulate P @ V.
        # Naive implementation: acc[i,d] += sum_j(p[i,j] * v[j,d]).
        for j in range(BLOCK_N):
            p_j = p[:, j:j+1]  # [BLOCK_M, 1]
            v_j = v[j:j+1, :]  # [1, BLOCK_DMODEL]
            p_j_layout = gl.convert_layout(p_j, gl.SliceLayout(dim=1, parent=acc_layout))
            v_j_layout = gl.convert_layout(v_j, gl.SliceLayout(dim=0, parent=acc_layout))
            acc = acc + p_j_layout[:, None] * v_j_layout[None, :]

        # Update max.
        m_i = m_ij

    # Normalize by sum.
    acc = acc / l_i[:, None]

    # Store output.
    o_base = Out + off_z * stride_oz + off_h_q * stride_oh
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    o_mask = offs_m[:, None] < N_CTX_Q
    gl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_mask)

    # Store log-sum-exp.
    l_ptrs = L + off_z * HQ * N_CTX_Q + off_h_q * N_CTX_Q + offs_m
    l_mask = offs_m < N_CTX_Q
    lse = m_i / 1.44269504089 + gl.log2(l_i) / 1.44269504089  # Convert from log2 to ln.
    gl.store(l_ptrs, lse, mask=l_mask)


# Full interface wrapper that matches attn_fwd signature.
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
                   USE_P_SCALE: gl.constexpr, INT8_KV: gl.constexpr):
    """
    Gluon Flash Attention Forward - Full Interface.

    This wrapper has the same signature as attn_fwd but delegates to the simple implementation.
    Currently only supports basic attention (no VARLEN, INT8, dropout, etc.).
    """
    # Validate unsupported features.
    gl.static_assert(not VARLEN, "VARLEN not supported in Gluon implementation")
    gl.static_assert(not INT8, "INT8 not supported in Gluon implementation")
    gl.static_assert(not ENABLE_DROPOUT, "Dropout not supported in Gluon implementation")
    gl.static_assert(not USE_ALIBI, "ALiBi not supported in Gluon implementation")
    gl.static_assert(not USE_BIAS, "Bias not supported in Gluon implementation")
    gl.static_assert(not PERSISTENT, "Persistent mode not supported in Gluon implementation")

    # Get program IDs.
    start_m = gl.program_id(1)
    off_z = gl.program_id(2)
    off_h_q = gl.program_id(0)

    # MQA/GQA head mapping.
    off_h_k = off_h_q * HK // HQ

    # Number of warps (fixed for now).
    NUM_WARPS: gl.constexpr = 4

    # Create 2D blocked layout for tiles.
    acc_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, BLOCK_DMODEL // 32 if BLOCK_DMODEL >= 32 else 1],
        threads_per_warp=[32, 1],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )

    # 1D layouts.
    offs_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=acc_layout)
    offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=acc_layout)

    # Offsets.
    offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=offs_m_layout)
    offs_d = gl.arange(0, BLOCK_DMODEL, layout=offs_d_layout)

    # Base pointers.
    q_base = Q + off_z * stride_qz + off_h_q * stride_qh
    k_base = K + off_z * stride_kz + off_h_k * stride_kh
    v_base = V + off_z * stride_vz + off_h_k * stride_vh

    # Load Q tile.
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = offs_m[:, None] < MAX_SEQLENS_Q
    if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
        q_mask = q_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    q = gl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize accumulators.
    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=offs_m_layout)
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=offs_m_layout)
    acc = gl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=gl.float32, layout=acc_layout)

    # Scale.
    qk_scale: gl.constexpr = SM_SCALE * 1.44269504089

    # Number of K/V blocks.
    n_blocks = (MAX_SEQLENS_K + BLOCK_N - 1) // BLOCK_N
    if IS_CAUSAL:
        n_blocks_causal = ((start_m + 1) * BLOCK_M + MAX_SEQLENS_K - MAX_SEQLENS_Q + BLOCK_N - 1) // BLOCK_N
        n_blocks = min(n_blocks, n_blocks_causal)

    # KV layout.
    kv_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, BLOCK_DMODEL // 32 if BLOCK_DMODEL >= 32 else 1],
        threads_per_warp=[32, 1],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kv_layout)

    # Main loop.
    for block_n in range(0, n_blocks):
        start_n = block_n * BLOCK_N
        offs_n = start_n + gl.arange(0, BLOCK_N, layout=offs_n_layout)

        # Load K.
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_mask = offs_n[:, None] < MAX_SEQLENS_K
        if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
            k_mask = k_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
        k = gl.load(k_ptrs, mask=k_mask, other=0.0)

        # Compute QK^T using tl.dot equivalent - convert layouts and use dot_fma.
        # For now, compute as outer product sum (naive but correct).
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=acc_layout)

        # Matrix multiply Q @ K^T.
        # q is [BLOCK_M, BLOCK_DMODEL], k is [BLOCK_N, BLOCK_DMODEL].
        # Result qk is [BLOCK_M, BLOCK_N].
        # qk[m, n] = sum_d q[m, d] * k[n, d].

        # Convert k to proper layout for multiplication.
        k_t = gl.convert_layout(k, kv_layout)  # [BLOCK_N, BLOCK_DMODEL]

        # Naive batched outer product.
        for d in range(0, BLOCK_DMODEL):
            q_slice = q[:, d]  # [BLOCK_M]
            k_slice = k_t[:, d]  # [BLOCK_N]
            # Outer product q_slice[:, None] * k_slice[None, :].
            q_col = gl.expand_dims(q_slice, 1)  # [BLOCK_M, 1]
            k_row = gl.expand_dims(k_slice, 0)  # [1, BLOCK_N]
            qk = qk + q_col * k_row

        # Scale.
        qk = qk * qk_scale

        # Causal mask.
        if IS_CAUSAL:
            causal_offs = start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=acc_layout))
            causal_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(dim=1, parent=acc_layout))
            causal_bound = causal_offs[None, :] + MAX_SEQLENS_Q - MAX_SEQLENS_K
            c_mask = causal_m[:, None] >= causal_bound
            neg_inf = gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=acc_layout)
            qk = gl.where(c_mask, qk, neg_inf)

        # Boundary mask.
        bound_offs = start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=acc_layout))
        b_mask = bound_offs[None, :] < MAX_SEQLENS_K
        neg_inf2 = gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=acc_layout)
        qk = gl.where(b_mask, qk, neg_inf2)

        # Online softmax.
        m_ij = gl.reduce(qk, axis=1, fn="max")
        m_new = gl.where(m_ij > m_i, m_ij, m_i)
        p = gl.exp2(qk - m_new[:, None])
        l_ij = gl.reduce(p, axis=1, fn="sum")
        alpha = gl.exp2(m_i - m_new)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        m_i = m_new

        # Load V.
        v_ptrs = v_base + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
        v = gl.load(v_ptrs, mask=k_mask, other=0.0)
        v_t = gl.convert_layout(v, kv_layout)

        # Accumulate P @ V.
        # p is [BLOCK_M, BLOCK_N], v is [BLOCK_N, BLOCK_DMODEL].
        # acc[m, d] += sum_n p[m, n] * v[n, d].
        for n in range(0, BLOCK_N):
            p_slice = p[:, n]  # [BLOCK_M]
            v_slice = v_t[n, :]  # [BLOCK_DMODEL]
            p_col = gl.expand_dims(p_slice, 1)  # [BLOCK_M, 1]
            v_row = gl.expand_dims(v_slice, 0)  # [1, BLOCK_DMODEL]
            acc = acc + p_col * v_row

    # Normalize.
    acc = acc / l_i[:, None]

    # Store output.
    o_base = Out + off_z * stride_oz + off_h_q * stride_oh
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    o_mask = offs_m[:, None] < MAX_SEQLENS_Q
    if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
        o_mask = o_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    gl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_mask)

    # Store LSE.
    l_ptrs = L + off_z * HQ * MAX_SEQLENS_Q + off_h_q * MAX_SEQLENS_Q + offs_m
    l_mask = offs_m < MAX_SEQLENS_Q
    lse = m_i / 1.44269504089 + gl.log2(l_i) / 1.44269504089
    gl.store(l_ptrs, lse, mask=l_mask)
