"""
- call sub-kernels
- using tl.exp2()
- specific flash-attention
- tl.static_assert()
- multi-axis launch grid
- pre-compute in separate kernel
- using approximate constants

DOES NOT INCLUDE
- only fp32
- dropout
- non-causal
- i forgor 💀
"""

import torch
import triton
import triton.language as tl
import math
DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")

@triton.jit
def _attn_forward_inner(
    Q, O, L, M,
    K_ptr, V_ptr,
    K_T_offsets, V_offsets,
    block_index_QO,
    scale,
    stride_K_N, stride_V_N,
    N,
    BLOCK_SIZE_QO: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
    DIAGONAL: tl.constexpr,
    offsets_QO_N: tl.constexpr, offsets_KV_N: tl.constexpr,
    Dh: tl.constexpr
):
    if DIAGONAL:
        lo = block_index_QO * BLOCK_SIZE_QO
        hi = (block_index_QO + 1) * BLOCK_SIZE_QO
    else:
        lo, hi = 0, block_index_QO * BLOCK_SIZE_QO

    K_T_offsets += lo * stride_K_N
    V_offsets += lo * stride_V_N
    offsets_KV_N += lo

    for start_KV in range(lo, hi, BLOCK_SIZE_KV):
        start_KV = tl.multiple_of(start_KV, BLOCK_SIZE_KV)
        mask_KV_N = offsets_KV_N < N
        K_T = tl.load(K_ptr + K_T_offsets, mask=mask_KV_N[None, :], other=0.0) # shape (Dh, BLOCK_SIZE_KV)
        S = tl.dot(Q, K_T) * scale # shape (BLOCK_SIZE_QO, BLOCK_SIZE_KV)

        if DIAGONAL:
            causal_mask = offsets_QO_N[:, None] >= offsets_KV_N[None, :]
            S = tl.where(causal_mask, S, -1e6) # shape (BLOCK_SIZE_QO, BLOCK_SIZE_KV)

        M_new = tl.maximum(M, tl.max(S, axis=1)) # shape (BLOCK_SIZE_QO,)
        S -= M_new[:, None]

        P = tl.exp2(S) # shape (BLOCK_SIZE_QO, BLOCK_SIZE_KV)
        L_new = tl.sum(P, axis=1) # shape (BLOCK_SIZE_QO,)
        alpha = tl.exp2(M - M_new)
        L = L * alpha + L_new

        V = tl.load(V_ptr + V_offsets, mask=mask_KV_N[:, None], other=0.0) 
        O = O * alpha[:, None] # shape (BLOCK_SIZE_QO, Dh)
        O = tl.dot(P, V, acc=O)

        M = M_new
        K_T_offsets += BLOCK_SIZE_KV * stride_K_N
        V_offsets += BLOCK_SIZE_KV * stride_V_N
        offsets_KV_N += BLOCK_SIZE_KV
    
    return O, L, M

@triton.autotune(
        [
            triton.Config(
                {"BLOCK_SIZE_QO": BLOCK_SIZE_QO, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
                num_stages=num_stages, num_warps=num_warps
            )
            for BLOCK_SIZE_QO in [16, 32, 64, 128]
            for BLOCK_SIZE_KV in [16, 32, 64, 128]
            for num_stages in [3]#, 5, 7]
            for num_warps in [4,]#8, 16]
        ],
        key=["Dh"]
)

@triton.jit
def attn_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    LSE_ptr,
    scale,
    stride_Q_B, stride_Q_H, stride_Q_N, stride_Q_Dh,
    stride_K_B, stride_K_H, stride_K_N, stride_K_Dh,
    stride_V_B, stride_V_H, stride_V_N, stride_V_Dh,
    stride_O_B, stride_O_H, stride_O_N, stride_O_Dh,
    stride_LSE_B, stride_LSE_H, stride_LSE_N,
    B, H, N,
    Dh: tl.constexpr,
    BLOCK_SIZE_QO: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr
):
    rln2: tl.constexpr = 1.4426950408889634 # 1 / math.log(2)
    # e^x = 2^(x * rln2)
    scale *= rln2

    tl.static_assert(BLOCK_SIZE_KV <= Dh)

    index_BH = tl.program_id(axis=1)
    index_B = index_BH // H
    index_H = index_BH % H
    Q_ptr += index_B * stride_Q_B + index_H * stride_Q_H
    K_ptr += index_B * stride_K_B + index_H * stride_K_H
    V_ptr += index_B * stride_V_B + index_H * stride_V_H
    O_ptr += index_B * stride_O_B + index_H * stride_O_H
    
    block_index_QO = tl.program_id(axis=0)
    offsets_QO_N = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO)
    offsets_KV_N = tl.arange(0, BLOCK_SIZE_KV)
    offsets_Dh = tl.arange(0, Dh)

    Q_offsets = offsets_QO_N[:, None] * stride_Q_N + offsets_Dh[None, :] * stride_Q_Dh # shape (BLOCK_SIZE_QO, Dh)
    K_T_offsets = offsets_Dh[:, None] * stride_K_Dh + offsets_KV_N[None, :] * stride_K_N # shape (Dh, BLOCK_SIZE_KV)
    V_offsets = offsets_KV_N[:, None] * stride_V_N + offsets_Dh[None, :] * stride_V_Dh # shape (BLOCK_SIZE_KV, Dh)

    mask_QO_N = offsets_QO_N < N # if Dh was not multiple of warp size we would also mask
    Q = tl.load(Q_ptr + Q_offsets, mask=mask_QO_N[:, None], other=0.0) # shape (BLOCK_SIZE_QO, Dh)

    M = tl.full(shape=[BLOCK_SIZE_QO], value=-1e6, dtype=tl.float32)
    L = tl.full(shape=[BLOCK_SIZE_QO], value=1.0, dtype=tl.float32) # e^0 = 1
    O = tl.zeros((BLOCK_SIZE_QO, Dh), dtype=tl.float32)

    O, L, M = _attn_forward_inner(
        Q, O, L, M,
        K_ptr, V_ptr,
        K_T_offsets, V_offsets,
        block_index_QO,
        scale,
        stride_K_N, stride_V_N,
        N,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV,
        False, # diagonal
        offsets_QO_N, offsets_KV_N,
        Dh
    )

    O, L, M = _attn_forward_inner(
        Q, O, L, M,
        K_ptr, V_ptr,
        K_T_offsets, V_offsets,
        block_index_QO,
        scale,
        stride_K_N, stride_V_N,
        N,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV,
        True, # diagonal
        offsets_QO_N, offsets_KV_N,
        Dh
    )

    O = O / L[:, None]
    LSE = M + tl.math.log2(L) # shape (BLOCK_SIZE_QO,)
    """
    softmax(x_i) = exp(x_i - m_i) / l_i
                 = exp(x_i - m_i) / exp(log(l_i))
                 = exp(x_i - (m_i + log(l_i)))
    """
    LSE_offsets = index_BH * stride_LSE_H + offsets_QO_N
    LSE_mask = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO) < N
    tl.store(LSE_ptr + LSE_offsets, LSE, mask=LSE_mask)

    O_offsets = offsets_QO_N[:, None] * stride_O_N + offsets_Dh[None, :] * stride_O_Dh
    tl.store(O_ptr + O_offsets, O, mask=mask_QO_N[:, None])


@triton.autotune(
        [
            triton.Config(
                {"PRE_BLOCK_SIZE_ROW": PRE_BLOCK_SIZE_ROW},
                num_stages=num_stages, num_warps=num_warps
            )
            for PRE_BLOCK_SIZE_ROW in [16, 32, 64, 128]
            for num_stages in [3]#, 5, 7]
            for num_warps in [4,]#8, 16]
        ],
        key=["Dh"]
)
@triton.jit
def attn_backward_preprocess(
    O_ptr, dLdO_ptr, delta_ptr,
    stride_O_B, stride_O_H, stride_O_N, stride_O_Dh,
    stride_delta_B, stride_delta_H, stride_delta_N,
    N, Dh: tl.constexpr,
    PRE_BLOCK_SIZE_ROW: tl.constexpr
):
    index_BH = tl.program_id(axis=1)
    row = tl.program_id(axis=0)
    row_offsets = row * PRE_BLOCK_SIZE_ROW + tl.arange(0, PRE_BLOCK_SIZE_ROW)
    col_offsets = tl.arange(0, Dh)
    mask = row_offsets < N

    O_ptr += index_BH * stride_O_H
    O_offsets = row_offsets[:, None] * stride_O_N + col_offsets[None, :] * stride_O_Dh # shape (PRE_BLOCK_SIZE_ROW, Dh)
    O = tl.load(O_ptr + O_offsets, mask=mask[:, None], other=0.0) 

    dLdO_ptr += index_BH * stride_O_H
    dLdO_offsets = row_offsets[:, None] * stride_O_N + col_offsets[None, :] * stride_O_Dh
    dLdO = tl.load(dLdO_ptr + dLdO_offsets, mask=mask[:, None], other=0.0)

    delta = tl.sum(dLdO * O, axis=1) # size (PRE_BLOCK_SIZE_ROW,)
    delta_ptr += index_BH * stride_delta_H
    tl.store(delta_ptr + row_offsets, delta, mask=mask)

@triton.jit
def _attention_backward_Q(
    Q, dLdO, dLdQ, LSE,
    K_ptr, V_ptr, delta_ptr,
    stride_Q_N, stride_Q_Dh,
    H, N, Dh: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
    start_ROW, start_COL, num_steps,
    scale, ln2: tl.constexpr, rln2: tl.constexpr,
    MASK: tl.constexpr
):
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    offsets_Dh = tl.arange(0, Dh)

    K_and_V_T_offsets = offsets_Dh[:, None] * stride_Q_Dh + offsets_COL[None, :] * stride_Q_N
    delta = tl.load(delta_ptr + offsets_ROW, mask=offsets_ROW < N, other=0.0)

    for block_idx in range(num_steps):
        KV_mask = (offsets_COL < N)[None, :]
        K_T = tl.load(K_ptr + K_and_V_T_offsets, mask=KV_mask, other=0.0) # shape (Dh, BLOCK_SIZE_COL)
        V_T = tl.load(V_ptr + K_and_V_T_offsets, mask=KV_mask, other=0.0) 

        S = tl.dot(Q, K_T) # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
        P = tl.exp2(S - LSE) # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)

        if MASK:
            P = tl.where(offsets_ROW[:, None] >= offsets_COL[None, :], P, 0.)

        dLdP = tl.dot(dLdO, V_T) # shape (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
        dLdS = P * (dLdP - delta[:, None]) * ln2

        dLdQ = tl.dot(dLdS, tl.trans(K_T), acc=dLdQ) # shape (BLOCK_SIZE_ROW, Dh)

        offsets_COL += BLOCK_SIZE_COL
        K_ptr += BLOCK_SIZE_COL * stride_Q_N
        V_ptr += BLOCK_SIZE_COL * stride_Q_N

    return dLdQ


@triton.jit
def _attention_backward_KV(
    K, V, dLdK, dLdV,
    Q_ptr, dLdO_ptr, LSE_ptr, delta_ptr,
    stride_Q_N, stride_Q_Dh,
    H, N, Dh: tl.constexpr,
    BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
    start_ROW, start_COL, num_steps,
    scale, ln2, rln2,
    MASK: tl.constexpr
):
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    offsets_Dh = tl.arange(0, Dh)

    Q_T_offsets = offsets_Dh[:, None] * stride_Q_Dh + offsets_ROW[None, :] * stride_Q_N
    dLdO_offsets = offsets_ROW[:, None] * stride_Q_N + offsets_Dh[None, :] * stride_Q_Dh

    for block_idx in range(num_steps):
        mask_N = offsets_ROW < N
        Q_T = tl.load(Q_ptr + Q_T_offsets, mask=mask_N[None, :], other=0.0) # shape (Dh, BLOCK_SIZE_ROW)
        LSE = tl.load(LSE_ptr + offsets_ROW, mask=mask_N, other=0.0) # shape (BLOCK_SIZE_ROW)
        dLdO = tl.load(dLdO_ptr + dLdO_offsets, mask=mask_N[:, None], other=0.0) # shape (BLOCK_SIZE_ROW, Dh)
        delta = tl.load(delta_ptr + offsets_ROW, mask=mask_N, other=0.0) # shape (BLOCK_SIZE_ROW,)

        S_T = tl.dot(K, Q_T) # shape (BLOCK_SIZE_COL, BLOCK_SIZE_ROW)
        P_T = tl.exp2(S_T - LSE[None, :]) # shape (BLOCK_SIZE_COL, BLOCK_SIZE_ROW)

        if MASK:
            mask = (offsets_COL[:, None] <= offsets_ROW[None, :])
            P_T = tl.where(mask, P_T, 0.)

        dLdV = tl.dot(P_T, dLdO, acc=dLdV)
        dLdP_T = tl.dot(V, tl.trans(dLdO)) # shape (BLOCK_SIZE_COL, BLOCK_SIZE_ROW)
        dLdS_T = (P_T * (dLdP_T - delta[None, :]) * ln2) # shape (BLOCK_SIZE_COL, BLOCK_SIZE_ROW)
        dLdK = tl.dot(dLdS_T, tl.trans(Q_T), acc=dLdK) # shape (BLOCK_SIZE_COL, Dh)

        offsets_ROW += BLOCK_SIZE_ROW
        Q_ptr += BLOCK_SIZE_ROW * stride_Q_N
        dLdO_ptr += BLOCK_SIZE_ROW * stride_Q_N

    return dLdK, dLdV

@triton.autotune(
        [
            triton.Config(
                {"BLOCK_SIZE_MACRO": BLOCK_SIZE_MACRO, "BLOCK_SIZE_MICRO": BLOCK_SIZE_MICRO},
                num_stages=num_stages, num_warps=num_warps
            )
            for BLOCK_SIZE_MICRO in [16, 32, 64]
            for BLOCK_SIZE_MACRO in [32, 64]#, 128]
            for num_stages in [3]#, 5, 7]
            for num_warps in [4,]#8, 16]
            if BLOCK_SIZE_MICRO < BLOCK_SIZE_MACRO
        ],
        key=["Dh"]
)
@triton.jit
def attn_backward(
    Q_ptr, K_ptr, V_ptr,
    dLdO_ptr, dLdQ_ptr, dLdK_ptr, dLdV_ptr,
    LSE_ptr, delta_ptr,
    scale,
    stride_Q_B, stride_Q_H, stride_Q_N, stride_Q_Dh,
    H, N, Dh: tl.constexpr,
    BLOCK_SIZE_MACRO: tl.constexpr, BLOCK_SIZE_MICRO: tl.constexpr
):
    ln2: tl.constexpr = 0.6931471824645996
    rln2: tl.constexpr = 1.4426950408889634

    idx_BH = tl.program_id(axis=1)
    idx_B = idx_BH // H
    idx_H = idx_BH % H

    BH_jump = idx_B * stride_Q_B + idx_H * stride_Q_H
    Q_ptr += BH_jump
    K_ptr += BH_jump
    V_ptr += BH_jump
    dLdO_ptr += BH_jump
    dLdQ_ptr += BH_jump
    dLdK_ptr += BH_jump
    dLdV_ptr += BH_jump

    BH_jump = idx_BH * N # LSE has shape (B, H, N), so LSE.stride() == (H*N, N, 1); LSE.stride(1) == N
    LSE_ptr += BH_jump
    delta_ptr += BH_jump

    tl.static_assert(BLOCK_SIZE_MACRO % BLOCK_SIZE_MICRO == 0)

    # dLdK and dLdV
    BLOCK_SIZE_ROW_1: tl.constexpr = BLOCK_SIZE_MICRO
    BLOCK_SIZE_COL_1: tl.constexpr = BLOCK_SIZE_MACRO

    pid = tl.program_id(axis=0)
    start_COL = pid * BLOCK_SIZE_COL_1
    start_ROW = start_COL
    num_steps = BLOCK_SIZE_COL_1 // BLOCK_SIZE_ROW_1

    offsets_COL_1 = start_COL + tl.arange(0, BLOCK_SIZE_COL_1)
    offsets_Dh = tl.arange(0, Dh)
    KV_offsets = offsets_COL_1[:, None] * stride_Q_N + offsets_Dh[None, :] * stride_Q_Dh
    KV_mask = offsets_COL_1[:, None] < N
    K = tl.load(K_ptr + KV_offsets, mask=KV_mask, other=0.0) 
    V = tl.load(V_ptr + KV_offsets, mask=KV_mask, other=0.0) # shape (BLOCK_SIZE_COL_1, Dh)

    K *= scale * rln2

    dLdK = tl.zeros((BLOCK_SIZE_COL_1, Dh), dtype=tl.float32)
    dLdV = tl.zeros((BLOCK_SIZE_COL_1, Dh), dtype=tl.float32)

    dLdK, dLdV = _attention_backward_KV(
        K, V, dLdK, dLdV,
        Q_ptr, dLdO_ptr, LSE_ptr, delta_ptr,
        stride_Q_N, stride_Q_Dh,
        H, N, Dh,
        BLOCK_SIZE_ROW_1, BLOCK_SIZE_COL_1,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=True
    )

    start_ROW += BLOCK_SIZE_MACRO
    N_adj = tl.cdiv(N, BLOCK_SIZE_COL_1) * BLOCK_SIZE_COL_1
    num_steps = (N_adj - start_ROW) // BLOCK_SIZE_MICRO

    dLdK, dLdV = _attention_backward_KV(
        K, V, dLdK, dLdV,
        Q_ptr, dLdO_ptr, LSE_ptr, delta_ptr,
        stride_Q_N, stride_Q_Dh,
        H, N, Dh,
        BLOCK_SIZE_ROW_1, BLOCK_SIZE_COL_1,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=False
    )

    dLdK *= scale * rln2
    tl.store(dLdK_ptr + KV_offsets, dLdK, mask=KV_mask)
    tl.store(dLdV_ptr + KV_offsets, dLdV, mask=KV_mask)

    # dLdQ
    BLOCK_SIZE_ROW_2: tl.constexpr = BLOCK_SIZE_MACRO
    BLOCK_SIZE_COL_2: tl.constexpr = BLOCK_SIZE_MICRO

    start_ROW = pid * BLOCK_SIZE_ROW_2
    start_COL = start_ROW
    num_steps = BLOCK_SIZE_ROW_2 // BLOCK_SIZE_COL_2

    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW_2)
    QO_offsets = offsets_ROW[:, None] * stride_Q_N + offsets_Dh[None, :] * stride_Q_Dh
    mask_ROW = offsets_ROW < N
    Q = tl.load(Q_ptr + QO_offsets, mask=mask_ROW[:, None], other=0.0)
    Q *= scale * rln2
    dLdO = tl.load(dLdO_ptr + QO_offsets, mask=mask_ROW[:, None], other=0.0)
    LSE = tl.load(LSE_ptr + offsets_ROW, mask=mask_ROW, other=0.0)[:, None]

    dLdQ = tl.zeros([BLOCK_SIZE_ROW_2, Dh], dtype=tl.float32)
    
    dLdQ = _attention_backward_Q(
        Q, dLdO, dLdQ, LSE,
        K_ptr, V_ptr, delta_ptr,
        stride_Q_N, stride_Q_Dh,
        H, N, Dh,
        BLOCK_SIZE_ROW_2, BLOCK_SIZE_COL_2,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=True
    )

    end_COL = start_COL
    start_COL = 0
    num_steps = end_COL // BLOCK_SIZE_COL_2

    dLdQ = _attention_backward_Q(
        Q, dLdO, dLdQ, LSE,
        K_ptr, V_ptr, delta_ptr,
        stride_Q_N, stride_Q_Dh,
        H, N, Dh,
        BLOCK_SIZE_ROW_2, BLOCK_SIZE_COL_2,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=False
    )

    dLdQ *= scale * rln2
    tl.store(dLdQ_ptr + QO_offsets, dLdQ, mask=mask_ROW[:, None])


class _flashattention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale):
        assert q.shape == k.shape == v.shape, "q, k, v must have the same shape"
        assert q.shape[-1] <= 128, "kernel only supports head dimension up to 128" # I'll do math on 1660 Super SRAM later
        assert q.device == k.device == v.device, "q, k, v must be on the same device"
        assert q.dtype == k.dtype == v.dtype == torch.float32, "q, k, v must be float32"
        B, H, N, Dh = q.shape

        O = torch.empty_like(q)
        LSE = torch.empty((B, H, N), device=q.device, dtype=torch.float32)

        grid = lambda args: (
            triton.cdiv(N, args["BLOCK_SIZE_QO"]),
            B * H
        )

        attn_fwd[grid](
            q, k, v, O, LSE, scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), q.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            B, H, N, Dh,

        )

        ctx.save_for_backward(q, k, v, O, LSE)
        ctx.grid = grid
        ctx.B, ctx.H, ctx.N, ctx.Dh = B, H, N, Dh
        ctx.scale = scale
        return O
    
    @staticmethod
    def backward(ctx, dLdO):
        q, k, v, O, LSE = ctx.saved_tensors
        grid = ctx.grid
        B, H, N, Dh = ctx.B, ctx.H, ctx.N, ctx.Dh
        scale = ctx.scale

        dLdq = torch.empty_like(q)
        dLdk = torch.empty_like(k)
        dLdv = torch.empty_like(v)

        dLdO = dLdO.contiguous()
        assert q.stride() == k.stride() == v.stride() == O.stride() == dLdO.stride()

        delta = torch.empty_like(LSE) # shape (B, H, N)
        pre_grid = lambda meta: (triton.cdiv(N, meta["PRE_BLOCK_SIZE_ROW"]), B * H)
        attn_backward_preprocess[pre_grid](
            O, dLdO, delta,
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            delta.stride(0), delta.stride(1), delta.stride(2),
            N, Dh
        )

        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_MACRO"]), B * H)
        attn_backward[grid](
            q, k, v,
            dLdO, dLdq, dLdk, dLdv,
            LSE, delta,
            scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            H, N, Dh
        )

        return dLdq, dLdk, dLdv, None

flash_attention = _flashattention.apply

def test_flash_attention_forward(B, H, N, Dh, device=DEVICE, atol=5e-3):
    q = torch.randn((B, H, N, Dh), device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn((B, H, N, Dh), device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn((B, H, N, Dh), device=device, dtype=torch.float32, requires_grad=True)
    scale = 1 / math.sqrt(Dh)

    # forward
    attention_triton = flash_attention(q, k, v, scale)
    attention_torch = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(attention_triton, attention_torch, atol=atol, rtol=0)
    print(f"Forward test passed for B={B}, H={H}, N={N}, Dh={Dh}")

def test_flash_attention_backward(B, H, N, Dh, device=DEVICE, atol=5e-3):
    q = torch.randn((B, H, N, Dh), device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn((B, H, N, Dh), device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn((B, H, N, Dh), device=device, dtype=torch.float32, requires_grad=True)
    scale = 1 / math.sqrt(Dh)

    # forward
    attention_triton = flash_attention(q, k, v, scale)
    attention_torch = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    
    # backward
    dLdO = 0.1 * torch.randn_like(q)
    attention_triton.backward(dLdO, retain_graph=True)
    dLdq_triton, dLdk_triton, dLdv_triton = [tensor.grad.clone() for tensor in [q, k, v]]
    q.grad, k.grad, v.grad = None, None, None
    attention_torch.backward(dLdO, retain_graph=True)
    dLdq_torch, dLdk_torch, dLdv_torch = [tensor.grad.clone() for tensor in [q, k, v]]
    torch.testing.assert_close(dLdq_triton, dLdq_torch, atol=atol, rtol=0)
    torch.testing.assert_close(dLdk_triton, dLdk_torch, atol=atol, rtol=0)
    torch.testing.assert_close(dLdv_triton, dLdv_torch, atol=atol, rtol=0)
    print(f"Backward test passed for B={B}, H={H}, N={N}, Dh={Dh}")

configs = [
    triton.testing.Benchmark(
        x_names=["SEQ_LEN"],
        x_vals = [512 * i for i in range(1, 17)],
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["torch.nn.functional.scaled_dot_product_attention", "our flash_attention"],
        styles=[("red", '-'), ("blue", '-')],
        ylabel="TFLOPS",
        plot_name=f"attention-performance-{mode}",
        args={"mode": mode}
    )
    for mode in ["fwd", "bwd"]
]

@triton.testing.perf_report(configs)
def benchmark_flash_attention(SEQ_LEN, mode, provider, device=DEVICE):
    assert mode in ["fwd", "bwd"]
    dtype = torch.float32
    BATCH, N_HEADS = 32, 4
    HEAD_DIM = 64 # i think 128 is too big for my 1660
    q = torch.randn((BATCH, N_HEADS, SEQ_LEN, HEAD_DIM), device=device, dtype=dtype, requires_grad=True)
    k = torch.randn((BATCH, N_HEADS, SEQ_LEN, HEAD_DIM), device=device, dtype=dtype, requires_grad=True)
    v = torch.randn((BATCH, N_HEADS, SEQ_LEN, HEAD_DIM), device=device, dtype=dtype, requires_grad=True)
    scale = 1 / math.sqrt(HEAD_DIM)
    if provider == "torch":
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    elif provider == "triton":
        fn = lambda: flash_attention(q, k, v, scale)
    if mode == "bwd":
        O = fn()
        dLdO = torch.randn_like(O)
        fn = lambda: O.backward(dLdO, retain_graph=True)
    ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * N_HEADS * SEQ_LEN * SEQ_LEN * HEAD_DIM
    total_flops = flops_per_matmul
    if mode == "bwd":
        total_flops *= 2.5
    return total_flops * 1e-12 / (ms * 1e-3)  # TFLOPS

if __name__ == "__main__":

    test_flash_attention_forward(1, 1, 128, 32)
    test_flash_attention_forward(1, 1, 128, 64)
    test_flash_attention_forward(1, 1, 128, 128)
    test_flash_attention_forward(32, 8, 69, 128)

    test_flash_attention_backward(1, 1, 128, 32)
    test_flash_attention_backward(1, 1, 128, 64)
    test_flash_attention_backward(1, 1, 128, 128)
    test_flash_attention_backward(32, 8, 69, 128)

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark_flash_attention.run(save_path='.', print_data=True)
