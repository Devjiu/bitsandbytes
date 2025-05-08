import math
import statistics

import torch

from bitsandbytes import functional as F
import triton
import triton.language as tl

from bitsandbytes.backends import xpu

torch.set_printoptions(precision=5, sci_mode=False, linewidth=120, edgeitems=20, threshold=10000)


def _quantile(a, q):
    n = len(a)
    a = sorted(a)

    def get_quantile(q):
        if not (0 <= q <= 1):
            raise ValueError("Quantiles must be in the range [0, 1]")
        point = q * (n - 1)
        lower = math.floor(point)
        upper = math.ceil(point)
        t = point - lower
        return (1 - t) * a[lower] + t * a[upper]

    return [get_quantile(q) for q in q]


def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = _quantile(times, quantiles)
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times
    elif return_mode == "min":
        return min(times)
    elif return_mode == "max":
        return max(times)
    elif return_mode == "mean":
        return statistics.mean(times)
    elif return_mode == "median":
        return statistics.median(times)


def measure_gpu(fn, *args):
    di = torch.xpu
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    cache_size = 256 * 1024 * 1024
    cache = torch.empty(int(cache_size // 4), dtype=torch.int, device="xpu")

    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn(*args)
    end_event.record()
    di.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5
    # compute number of warmup and repeat
    n_warmup = max(1, int(25 / estimate_ms))
    n_repeat = max(1, int(100 / estimate_ms))
    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn(*args)
    # Benchmark
    for i in range(n_repeat):
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn(*args)
        end_event[i].record()
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    times_med = _summarize_statistics(times, None, "median")
    return times_med


quant_blocksize = 64
model = 18
hidden = 16
torch.manual_seed(42)
b = torch.rand(hidden, model, device="xpu").half()
qb, SB = F.quantize_4bit(b, blocksize=quant_blocksize, quant_type="nf4")
a = torch.rand(16, model, device="xpu").half()

qb = qb.to(device="xpu")
SB.code = SB.code.to(device="xpu")
SB.absmax = SB.absmax.to(device="xpu")
a = a.to(device="xpu")
b = b.to(device="xpu")

out = torch.empty(16, hidden, device="xpu").half()

# med_tm = measure_gpu(xpu.compiled_nf4_gemm, a, qb, out, False, False, SB)
# print("     med linear_nf4 (torch compile): ", med_tm, "ms")


def raw_triton_dequant(A, B, state):
    #  qa, SA, SA.absmax, out_B, SA.blocksize, "nf4"
    out_B = torch.empty(state.shape, dtype=torch.float16, device=B.device)
    # B - > dqB (32, 16) -> (32, 8)
    dqB = xpu.dequant_nf4_fp16(B, state, state.absmax, out_B, state.blocksize, "nf4")
    output = torch.matmul(A, dqB.to(A.dtype).t())
    output = output.to(torch.float16)
    return output


out_ref = raw_triton_dequant(a, qb, SB)


@triton.jit
def dequant_4bit_kernel(a, offsets, quant_ptr, absmax_ptr, num_paired_elements, QUANT_BLOCK: tl.constexpr):
    PAIRED_QUANT_BLOCK = QUANT_BLOCK // 2
    a = a.to(tl.uint8, bitcast=True)

    # higher 4bits from uint8 packed tensor
    higher = a & 0xF
    # lower 4bits
    lower = a >> 4

    # apply conversion
    higher_nf4 = tl.load(quant_ptr + higher)
    lower_nf4 = tl.load(quant_ptr + lower)

    abs_blocks_lim = (
        num_paired_elements // PAIRED_QUANT_BLOCK
    ) * PAIRED_QUANT_BLOCK + num_paired_elements % PAIRED_QUANT_BLOCK
    abs_offsets = offsets // PAIRED_QUANT_BLOCK
    mask_blocked = offsets < abs_blocks_lim
    absmax = tl.load(absmax_ptr + abs_offsets, mask_blocked)
    # absmax = absmax.to(tl.float16)

    # apply scales
    mul_high = higher_nf4 * absmax
    mul_low = lower_nf4 * absmax

    out_dq = tl.interleave(mul_low, mul_high)
    return out_dq


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    quant_ptr,
    absmax_ptr,
    num_paired_elements,
    QUANT_BLOCK: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_bk = tl.arange(0, BLOCK_SIZE_K // 2)
    # A - [42, 300] B - [400, 300], причем по факту [60000, 1] то есть типа [400, 150] или [300, 200]
    # то есть из B нужно получить кусок типа [400, 150] деквантизовать и транспонировать
    # пусть B [4, 6]
    # [ (1,  2),  (3,  4),  (5,  6),
    #   (7,  8),  (9, 10), (11, 12),
    #  (13, 14),  (15, 16), (17, 18),
    #  (19, 20),  (21, 22), (23, 24) ] типа [4, 3] ([4, 6])

    # транспонированный:
    # [ 1,  7, 13, 19,
    #   2,  8, 14, 20,
    #   3,  9, 15, 21,
    #   4, 10, 16, 22,
    #   5, 11, 17, 23,
    #   6, 12, 18, 24 ] типа [6, 4]

    # транспонированные гранулы:
    # [ (1, 2),  (7,  8), (13, 14), (19, 20),
    #   (3, 4),  (9, 10), (15, 16), (21, 22),
    #   (5, 6), (11, 12), (17, 18), (23, 24) ] тут [3, 4]

    # для A пусть размер [3, 6]
    # [ 1,  2,  3,  4,  5,  6,
    #   7,  8,  9, 10, 11, 12,
    #  13, 14, 15, 16, 17, 18 ] типа [3, 6]

    # возьмем типа [2, 3] размер блока A и соответственно [3, 2] B
    # тогда для транс B нужны куски [3, 2]:
    # [ 1,  7,    [ 13, 19,
    #   2,  8,      14, 20,
    #   3,  9 ]     15, 21 ]
    # [ 4, 10,      16, 22,
    #   5, 11,      17, 23,
    #   6, 12 ]     18, 24 ]

    # транспонированный кусок B:
    # [ 1,  2,  3,  [ 13, 14, 15,
    #   7,  8,  9]    19, 20, 21]
    # [ 4,  5,  6,  [ 16, 17, 18,
    #  10, 11, 12]    22, 23, 24]
    # некоторые гранулы разбиты в конце матрицы, некоторые в начале - пример гранула (3, 4)
    #
    # пойдем от обратного:
    # для B нужны 4 куска [3, 2] из [6, 4] (K, N)
    # [ 0_1,  0_2,    [ 1_1, 1_2,
    #   0_3,  0_4,      1_3, 1_4,
    #   0_5,  0_6 ]     1_5, 1_6 ]
    #
    # [ 2_1, 2_2,     [ 3_1, 3_2,
    #   2_3, 2_4,       3_3, 3_4,
    #   2_5, 2_6 ]      3_5, 3_6 ]

    # транспонированный полный B: (N, K) - блоки (b_N, b_K)
    # [ 0_1, 0_3, 0_5, 2_1, 2_3, 2_5,
    #   0_2, 0_4, 0_6, 2_2, 2_4, 2_6,
    #   1_1, 1_3, 1_5, 3_1, 3_3, 3_5,
    #   1_2, 1_4, 1_6, 3_2, 3_4, 3_6 ]

    # исходный гранулированный B: (N, K/2) - блоки (b_N, b_K/2) + off
    # [ (0_1, 0_3),  (0_5, 2_1),  (2_3, 2_5),
    #   (0_2, 0_4),  (0_6, 2_2),  (2_4, 2_6),
    #   (1_1, 1_3),  (1_5, 3_1),  (3_3, 3_5),
    #   (1_2, 1_4),  (1_6, 3_2),  (3_4, 3_6) ]

    # нужно загружать гранулы и учитывать глобальный оффсет

    # короче - нужно поддержать нечетный K и есть проблема _с_ остатоком от деления на block_K

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_offsets = offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    dq_b = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float16)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        b_ptrs = b_ptr + b_offsets
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        print("border: ", K // 2 - k * BLOCK_SIZE_K // 2)
        print("offs bk: ", offs_bk[None, :])
        print("mask: ", (offs_bk[None, :] < K // 2 - k * BLOCK_SIZE_K // 2))
        a = tl.load(a_ptrs, mask=((offs_k[None, :] < K - k * BLOCK_SIZE_K) & (offs_bn[:, None] < N)), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_bk[None, :] < K // 2 - k * BLOCK_SIZE_K // 2), other=0x77)
        print("loaded b: ", b)
        dq_b_t = dequant_4bit_kernel(
            b,
            b_offsets,
            quant_ptr,
            absmax_ptr,
            num_paired_elements,
            QUANT_BLOCK,
        )
        dq_b_t = dq_b_t.trans()
        dq_b = dq_b_t.to(tl.float16)

        # We accumulate along the K dimension.
        accumulator = tl.dot(a, dq_b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_offsets += (BLOCK_SIZE_K // 2) * stride_bk
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b, state):
    # Check constraints.
    # assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    N, K = state.shape

    # b_padded = torch.zeros((N, K//2 + K%2), device=a.device, dtype=torch.uint8)
    # if b.numel() % b_padded.numel() != 0:
    #     b_padded = b_padded.view(-1)
    #     b_padded[:b.numel()] = b.view(-1)
    #     b_padded = b_padded.view(N, K//2 + K%2)
    # b = b_padded
    # print("b shape: ", b.shape, " K + rest: ", K//2 + K%2)
    b = b.view(N, K // 2)
    b = b.to(torch.uint8)

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = 16, 16, 16, 4
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    print("c shape: M - ", M, " N - ", N, " K - ", K)
    # 1D launch kernel where each block gets its own program.
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    print("grid: M - ", triton.cdiv(M, BLOCK_SIZE_M), " N - ", triton.cdiv(N, BLOCK_SIZE_N))
    print("state code: ", state.code, " dtype: ", state.code.dtype)
    number_of_paired_elements = b.numel()
    matmul_kernel[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(1),
        b.stride(0),  #
        c.stride(0),
        c.stride(1),  #
        state.code,
        state.absmax,
        number_of_paired_elements,
        state.blocksize,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,  #
        GROUP_SIZE_M,
    )
    return c


ref_res = matmul(a, qb, SB)
print("out ref dtype: ", out_ref.dtype, " shape: ", out_ref.shape)

print("res ref dtype: ", ref_res.dtype, " shape: ", ref_res.shape)
mask = torch.isclose(ref_res, out_ref, atol=1e-2)
mask = ~mask
ranged = torch.arange(0, ref_res.numel()).view(ref_res.shape)
print("allclose mask: ", ranged[mask.cpu()][:40])
assert torch.allclose(ref_res, out_ref, atol=1e-2), "Output mismatch\n"
