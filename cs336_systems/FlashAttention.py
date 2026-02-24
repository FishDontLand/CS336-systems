import triton
import triton.language as tl
import torch
import einx
import math


class PyTorchFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.tensor, k: torch.tensor, v:torch.tensor, is_casual:bool):
        B_q = 16
        B_k = 16
        num_q_tiles = q.size(-2) // B_q
        num_k_tiles = k.size(-2) // B_k

        output = torch.empty(q.size(), device=q.device, dtype=q.dtype)
        logsumexp = torch.empty(q.size()[:-1], device=q.device, dtype=q.dtype)


        assert num_q_tiles * B_q == q.size(-2), "out of boundary issue!"
        assert num_k_tiles * B_k == k.size(-2), "out of boundary issue!"
        assert num_k_tiles * B_q == v.size(-2), "out of boundary issue!"

        for i in range(num_q_tiles):
            q_tile = q[..., i * B_q:(i + 1) * B_q, :]
            output_tile = torch.zeros_like(q_tile)
            prefix_sum = torch.zeros(q_tile.size()[:-1], dtype=q_tile.dtype, device=q_tile.device)
            prefix_max = torch.full(q_tile.size()[:-1], -float('inf'), dtype=q_tile.dtype, device=q_tile.device)
            for j in range(num_k_tiles):
                k_tile = k[..., j * B_k:(j + 1) * B_k, :]
                v_tile = v[..., j * B_k:(j + 1) * B_k, :]
                pre_softmax_tile = einx.dot('... b_q d, ... b_k d ->... b_q b_k', q_tile, k_tile) / math.sqrt(q.size(-1))
                new_prefix_max = torch.maximum(prefix_max, pre_softmax_tile.max(axis=-1)[0])
                softmax_tile = torch.exp(einx.subtract("... b_q b_k, ... b_q -> ... b_q b_k", pre_softmax_tile, new_prefix_max))
                max_adjust_fac = torch.exp(prefix_max - new_prefix_max)
                prefix_sum = max_adjust_fac * prefix_sum + softmax_tile.sum(axis=-1)
                output_tile = einx.multiply('... b_q, ... b_q d -> ... b_q d', max_adjust_fac, output_tile) + \
                              einx.dot("... b_q b_k, ... b_k d -> ... b_q d", softmax_tile, v_tile)
                prefix_max = new_prefix_max

            output[..., i * B_q:(i + 1) * B_q, :] = einx.divide('... b_q d, ... b_q -> ... b_q d', output_tile, prefix_sum)
            logsumexp[..., i * B_q:(i + 1) * B_q] = prefix_max + torch.log(prefix_sum)

        ctx.save_for_backward(q, k, v, logsumexp)
        return output

    @staticmethod
    @torch.compile
    def backward(ctx, dO):
        Q, K, V, L = ctx.saved_tensors
        scale = 1 / math.sqrt(Q.size(-1))
        S = einx.dot('... q d, ... k d -> ... q k', Q, K) * scale
        P = torch.exp(einx.subtract('... q k, ... q -> ... q k', S, L))
        dV = einx.dot('... q k, ... q d -> ... k d', P, dO)
        dP = einx.dot('... q d, ... k d -> ... q k', dO, V)
        D = (P * dP).sum(axis=-1)
        dS = P * (einx.subtract('... q k, ... q -> ... q k', dP, D))
        dQ = einx.dot('... q k, ... k d -> ... q d', dS, K) * scale
        dK = einx.dot('... q k, ... q d -> ... k d', dS, Q) * scale
        return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_casual: tl.constexpr
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0)
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq, ),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    output_tile = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    prefix_sum = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    prefix_max = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)

    q_tile = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    if is_casual:
        q_index_vec = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_tile = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        v_tile = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        pre_softmax_tile = tl.dot(q_tile, tl.trans(k_tile)) * scale

        if is_casual:
            k_index_vec = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            is_greater = q_index_vec[:, None] >= k_index_vec[None, :]
            pre_softmax_tile = tl.where(is_greater, pre_softmax_tile, -1e6)

        new_prefix_max = tl.maximum(prefix_max, tl.max(pre_softmax_tile, axis=-1))
        scaled_attn_tile = tl.exp(pre_softmax_tile - new_prefix_max[:, None])
        prefix_adj_fac = tl.exp(prefix_max - new_prefix_max)
        prefix_sum = prefix_adj_fac * prefix_sum + tl.sum(scaled_attn_tile, axis=-1)
        scaled_attn_tile = scaled_attn_tile.to(V_block_ptr.type.element_ty)
        output_tile = tl.dot(scaled_attn_tile, v_tile, acc=prefix_adj_fac[:, None] * output_tile)
        prefix_max = new_prefix_max

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))


    tl.store(O_block_ptr, (output_tile / prefix_sum[:, None]).to(V_block_ptr.type.element_ty), boundary_check=(0,))
    tl.store(L_block_ptr, prefix_max + tl.log(prefix_sum), boundary_check=(0,))

@triton.jit
def flash_bwd_kernel(
        Q_ptr, O_ptr, dO_ptr,
        K_ptr, V_ptr,
        L_ptr, D_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_ob, stride_oq, stride_od,
        stride_dob, stride_doq, stride_dod,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_lb, stride_lq,
        stride_db, stride_dq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_casual: bool,
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq, ),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    K_tile = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
    V_tile = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
    dK_tile = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_tile = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        Q_tile = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
        O_tile = tl.load(O_block_ptr, boundary_check=(0,), padding_option="zero")
        dO_tile = tl.load(dO_block_ptr, boundary_check=(0,), padding_option="zero")

    # remain to be done
    pass








class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.tensor, k: torch.tensor, v: torch.tensor, is_casual: bool):
        batch_size = q.size(0)
        q_len = q.size(-2)
        k_len = k.size(-2)
        v_len = v.size(-2)

        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16
        ctx.scale = math.sqrt(1 / q.size(-1))
        ctx.is_casual = is_casual

        NUM_Q_TILES = q_len // ctx.Q_TILE_SIZE
        NUM_K_TILES = k_len // ctx.K_TILE_SIZE

        assert NUM_Q_TILES * ctx.Q_TILE_SIZE == q_len, "query length must be divisible by tile size"
        assert NUM_K_TILES * ctx.K_TILE_SIZE == k_len, "key length must be divisible by tile size"
        assert k_len == v_len, "key length must be same as value length"

        output = torch.empty(q.size(),device=q.device)
        logsumexp = torch.empty(q.size()[:-1], device=q.device)

        flash_fwd_kernel[(NUM_Q_TILES, batch_size)](
            q, k, v,
            output, logsumexp,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            logsumexp.stride(0), logsumexp.stride(1),
            q.size(1), k.size(1),
            ctx.scale,
            q.size(-1),
            ctx.Q_TILE_SIZE,
            ctx.K_TILE_SIZE,
            is_casual
        )

        ctx.save_for_backward(q, k, v, logsumexp)
        return output.view(q.shape)

    @staticmethod
    @torch.compile
    def backward(ctx, dO):
        pass


def _attention_and_lse(q, k, v, is_causal=False):
    from einops import einsum
    n_queries = q.shape[-2]
    n_keys = k.shape[-2]
    d = q.shape[-1]
    scale = 1 / (d ** 0.5)
    S = einsum(q, k, '... q d, ... k d -> ... q k') * scale
    if is_causal:
        S = torch.where(
            torch.arange(n_queries, device=S.device)[None, :, None] >= torch.arange(n_keys, device=S.device)[None, None, :],
            S,
            -1e6
        )
    P = torch.softmax(S, dim=-1)
    o = einsum(P, v, '... q k, ... k d -> ... q d')
    L = torch.logsumexp(S, dim=-1)
    return o, L

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = torch.randn(1, 16, 16, device=device, requires_grad=True)
    k = torch.randn(1, 16, 16, device=device, requires_grad=True)
    v = torch.randn(1, 16 ,16, device=device, requires_grad=True)
    do = torch.randn(1, 16, 16, device=device)

    x1 = PyTorchFlashAttention.apply(q, k, v, False)
    x2 = x1.backward(do)
    # x2 = TritonFlashAttention.apply(q, k, v, False)

    # x3 = _attention_and_lse(q, k, v, True)[0]
    # x4 = TritonFlashAttention.apply(q, k, v, True)
    print('finished')