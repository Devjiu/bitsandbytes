import math
import statistics
from typing import Literal, Optional

import torch

from bitsandbytes import functional as F
import torch.nn.functional as F_T
from bitsandbytes.utils import QuantState
import triton
import triton.language as tl

k = 20

torch.set_printoptions(precision=5, sci_mode=False, linewidth=120, edgeitems=20, threshold=10000)


@triton.jit
def dequant_4bit_kernel(
    a_ptr, c_ptr, quant_ptr, absmax_ptr, num_paired_elements, QUANT_BLOCK: tl.constexpr, SPLIT_SIZE: tl.constexpr
):
    PAIRED_QUANT_BLOCK = QUANT_BLOCK // 2

    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * SPLIT_SIZE
    offsets = block_start + tl.arange(0, SPLIT_SIZE)
    mask = offsets < num_paired_elements

    a = tl.load(a_ptr + offsets, mask)
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

    # apply scales
    mul_high = higher_nf4 * absmax
    mul_low = lower_nf4 * absmax

    out_dq = tl.interleave(mul_low, mul_high)

    out_block_start = pid * SPLIT_SIZE * 2
    offs = out_block_start + tl.arange(0, SPLIT_SIZE * 2)
    mask = offs < num_paired_elements * 2
    tl.store(c_ptr + offs, out_dq, mask)


def dequant_nf4_fp16(
    A_nf4: torch.Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    quant_blocksize: int = 64,
    quant_type: Literal["fp4", "nf4"] = "fp4",
):
    transpose = True if A_nf4.shape[0] == 1 else False
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    # A_nf4 = A_nf4.to(device=DEVICE)
    # out = out.to(device=DEVICE)
    # absmax = absmax.to(device=DEVICE)
    if A_nf4.dtype != torch.uint8:
        print("[Warning] Forcing conversion of {A_nf4.dtype} to uint8.")
        bytes_value = A_nf4.cpu().numpy().tobytes()
        A_nf4 = torch.frombuffer(bytes_value, dtype=torch.uint8).to(A_nf4.device)

    if quant_state is None:
        assert absmax is not None and out is not None

        quant_state = QuantState(
            absmax=absmax,
            shape=out.shape,
            dtype=out.dtype,
            blocksize=quant_blocksize,
            quant_type=quant_type,
        )
    else:
        absmax = quant_state.absmax

    quant_state_code = quant_state.code.to(device=DEVICE)

    if quant_type not in ["nf4"]:
        raise NotImplementedError(
            f"4-bit quantization data type {quant_state.quant_type} is not implemented for CPU/XPU."
        )

    if quant_state.nested:
        # raise NotImplementedError("Fuck")
        # print("Quant state nested: ", quant_state.state2)
        absmax_orig = absmax
        # absmax = dequant_8bit(absmax_orig, quant_state.offset, quant_state.state2)
        # print("absmax shape out: ", absmax.shape, " absmax in shape: ", absmax_orig.shape)
        assert quant_state.state2.quant_type == "int8"
        assert quant_state.offset.numel() == 1
        # absmax_out = torch.empty(absmax.shape, dtype=quant_state.state2.dtype, device=absmax.device)
        absmax = dequant_int8_fp16(
            absmax_orig,
            quant_state.offset,
            quant_state.state2,
            quant_state.state2.absmax,
            quant_state.state2.blocksize,
        )

    if out is None:
        out = torch.empty(quant_state.shape, dtype=quant_state.dtype, device=A_nf4.device)

    number_of_paired_elements = A_nf4.numel()

    SPLIT_SIZE = 512
    # grid = lambda META: (triton.cdiv(number_of_paired_elements, SPLIT_SIZE), )
    grid = (triton.cdiv(number_of_paired_elements, SPLIT_SIZE),)
    # print("split: ", split_size, " grid: ", grid)
    # start = time.time()
    dequant_4bit_kernel[grid](
        A_nf4, out, quant_state_code, absmax, number_of_paired_elements, quant_state.blocksize, SPLIT_SIZE
    )
    # print("dequant_kernel: ", (time.time() - start), "s")

    if transpose:
        print("Transposing!")
        out = out.t()

    return out


def matmul_get_configs():
    return [
        triton.Config({"SPLIT_ROW": SPLIT_ROW, "SPLIT_COL": SPLIT_COL, "grf_mode": grf}, num_stages=s, num_warps=w)
        for SPLIT_ROW in [16, 32, 64, 128, 256]
        for SPLIT_COL in [16, 32, 64, 128, 256]
        for s in [3, 4]
        for w in [32]
        for grf in ["large", "auto"]
    ]


# @triton.autotune(
#     configs=matmul_get_configs(),
#     key=["R", "C"],
# )
@triton.jit
def dequant_4bit_kernel_2d(
    a_ptr,
    c_ptr,
    quant_ptr,
    absmax_ptr,  #
    R,
    C,  #
    stride_a_row,
    stride_a_col,  #
    QUANT_BLOCK: tl.constexpr,  #
    SPLIT_ROW: tl.constexpr,  #
    SPLIT_COL: tl.constexpr,  #
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    start_r = pid_m * SPLIT_ROW
    start_c = pid_n * SPLIT_COL

    # offs_a_row = start_r + tl.arange(0, SPLIT_ROW)
    # offs_a_row = tl.where(offs_a_row < R, offs_a_row, 0)
    # offs_a_col = start_c + tl.arange(0, SPLIT_COL)
    # offs_a_col = tl.where(offs_a_col < C, offs_a_col, 0)
    # offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    # offs_a_row = tl.max_contiguous(tl.multiple_of(offs_a_row, SPLIT_ROW), SPLIT_ROW)
    # offs_a_col = tl.max_contiguous(tl.multiple_of(offs_a_col, SPLIT_COL), SPLIT_COL)

    offs_a_row = (start_r + tl.arange(0, SPLIT_ROW)) % R
    offs_a_col = (start_c + tl.arange(0, SPLIT_COL)) % C
    # a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # print("offs row: ", offs_a_row)
    # print("offs col: ", offs_a_col)
    # print("strides row: ", stride_a_row, " stride col: ", stride_a_col)
    # offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, SPLIT_COL), SPLIT_COL)
    # offs_k = tl.arange(0, BLOCK_SIZE_K)
    offsets = offs_a_row[:, None] * stride_a_row + offs_a_col[None, :] * stride_a_col
    # print("total offsets: ", offsets)
    a_ptrs = a_ptr + offsets

    # a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
    a = tl.load(a_ptrs)
    a = a.to(tl.uint8, bitcast=True)
    # print("loaded a: ", a)

    PAIRED_QUANT_BLOCK = QUANT_BLOCK // 2

    # higher 4bits from uint8 packed tensor
    higher = a & 0xF
    # lower 4bits
    lower = a >> 4

    # print("int4 higher: ", higher)
    # print("int4  lower: ", lower)

    # apply conversion
    higher_nf4 = tl.load(quant_ptr + higher)
    lower_nf4 = tl.load(quant_ptr + lower)

    # print("converted higher: ", higher_nf4)
    # print("converted lower: ", lower_nf4)

    num_paired_elements = R * C * 2
    abs_blocks_lim = (
        num_paired_elements // PAIRED_QUANT_BLOCK
    ) * PAIRED_QUANT_BLOCK + num_paired_elements % PAIRED_QUANT_BLOCK
    abs_offsets = offsets // PAIRED_QUANT_BLOCK
    # print("abs offsets: ", abs_offsets)
    # print("abs_limit: ", abs_blocks_lim)
    mask_blocked = offsets < abs_blocks_lim
    absmax = tl.load(absmax_ptr + abs_offsets, mask_blocked)

    # apply scales
    mul_high = higher_nf4 * absmax
    mul_low = lower_nf4 * absmax

    out_dq = tl.interleave(mul_low, mul_high)

    # print("interleaved out: ", out_dq)

    out_start_c = pid_n * SPLIT_COL * 2
    offs_a_col = out_start_c + tl.arange(0, SPLIT_COL * 2)
    offs_a_col = tl.where(offs_a_col < C * 2, offs_a_col, 0)
    # out_block_start = pid * SPLIT_SIZE * 2
    offsets = offs_a_row[:, None] * stride_a_row * 2 + offs_a_col[None, :] * stride_a_col
    # print("out offsets: ", offsets)
    # offs = out_block_start + tl.arange(0, SPLIT_SIZE * 2)
    mask = offsets < num_paired_elements * 2
    # c_mask = (offs_a_row[:, None] < R) & (offs_a_col[None, :] < C * 2)
    # tl.store(c_ptr + offs, out_dq, mask)
    tl.store(c_ptr + offsets, out_dq, mask)


def dequant2d_nf4_fp16(
    A_nf4: torch.Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    quant_blocksize: int = 64,
    quant_type: Literal["fp4", "nf4"] = "fp4",
):
    transpose = True if A_nf4.shape[0] == 1 else False
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    # A_nf4 = A_nf4.to(device=DEVICE)
    # out = out.to(device=DEVICE)
    # absmax = absmax.to(device=DEVICE)
    if A_nf4.dtype != torch.uint8:
        print("[Warning] Forcing conversion of {A_nf4.dtype} to uint8.")
        bytes_value = A_nf4.cpu().numpy().tobytes()
        A_nf4 = torch.frombuffer(bytes_value, dtype=torch.uint8).to(A_nf4.device)

    if quant_state is None:
        assert absmax is not None and out is not None

        quant_state = QuantState(
            absmax=absmax,
            shape=out.shape,
            dtype=out.dtype,
            blocksize=quant_blocksize,
            quant_type=quant_type,
        )
    else:
        absmax = quant_state.absmax

    quant_state_code = quant_state.code.to(device=DEVICE)

    if quant_type not in ["nf4"]:
        raise NotImplementedError(
            f"4-bit quantization data type {quant_state.quant_type} is not implemented for CPU/XPU."
        )

    if quant_state.nested:
        raise NotImplementedError("Fuck")
        # print("Quant state nested: ", quant_state.state2)
        absmax_orig = absmax
        # absmax = dequant_8bit(absmax_orig, quant_state.offset, quant_state.state2)
        # print("absmax shape out: ", absmax.shape, " absmax in shape: ", absmax_orig.shape)
        assert quant_state.state2.quant_type == "int8"
        assert quant_state.offset.numel() == 1
        absmax_out = torch.empty(absmax.shape, dtype=quant_state.state2.dtype, device=absmax.device)
        absmax = dequant_int8_fp16(
            absmax_orig,
            quant_state.offset,
            quant_state.state2,
            quant_state.state2.absmax,
            absmax_out,
            quant_state.state2.blocksize,
        )

    if out is None:
        out = torch.empty(quant_state.shape, dtype=quant_state.dtype, device=A_nf4.device)

    number_of_paired_elements = A_nf4.numel()
    SPLIT_ROW = 64
    SPLIT_COL = 32
    R, C = out.shape
    # print("full tensor shape: ", out.shape)
    C = C // 2

    # Triton autotuning for function dequant_4bit_kernel_2d finished after 46.07s;
    # SPLIT_ROW: 64, SPLIT_COL: 32, grf_mode: auto, num_warps: 32, num_ctas: 1, num_stages: 3,
    # grid = lambda META: (triton.cdiv(R, META["SPLIT_ROW"]), triton.cdiv(C, META["SPLIT_COL"]), )
    grid = (
        triton.cdiv(R, SPLIT_ROW),
        triton.cdiv(C, SPLIT_COL),
    )
    # print("split: ", split_size, " grid: ", grid)
    # start = time.time()
    # print("A_nf4 shape: ", A_nf4.shape, " stride: ", A_nf4.stride())
    # print("out shape: ", out.shape, " stride: ", out.stride())
    dequant_4bit_kernel_2d[grid](
        A_nf4,
        out,
        quant_state_code,
        absmax,
        R,
        C,
        C,
        out.stride(1),
        quant_state.blocksize,
        SPLIT_ROW=SPLIT_ROW,
        SPLIT_COL=SPLIT_COL,
        grf_mode="auto",
        num_warps=32,
        num_ctas=1,
        num_stages=3,
    )

    if transpose:
        print("Transposing!")
        out = out.t()

    return out

# def dequant_8bit(A, offset, quant_state):
def dequant_int8_fp16(
    A_nf4: torch.Tensor,
    bias: torch.Tensor,
    quant_state: QuantState,
    absmax: torch.Tensor,
    quant_blocksize: int = 64,
):
    assert A_nf4.dtype == torch.uint8
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    quant_state_code = quant_state.code.to(device=DEVICE)

    absmax = quant_state_code[A_nf4.reshape(-1).int()]
    # print("[ref] scaled A: ", absmax)
    blocks = absmax.shape[-1] // quant_blocksize
    res = absmax.shape[-1] % quant_blocksize
    if res != 0:
        absmax = F_T.pad(absmax, (0, quant_blocksize - res), mode="constant", value=0)
    absmax = (absmax.view(-1, quant_blocksize) * quant_state.absmax.view(-1, 1)).to(quant_state.dtype).reshape(-1)
    absmax = absmax[: blocks * quant_blocksize + res]
    absmax = absmax.reshape(A_nf4.shape)
    absmax += bias
    return absmax

def dequantize_nf4(
    a: torch.Tensor,
    quant_state: QuantState,
    out: torch.Tensor,
    quant_range: torch.Tensor,
    absmax: torch.Tensor,
    blocksize,
):
    a = a.reshape(-1)
    out_dq = torch.empty(a.size(0) * 2, dtype=torch.int32)
    n = out_dq.numel()
    # higher 4bits from uint8 packed tensor
    out_dq[1::2] = a & 0xF
    # lower 4bits
    out_dq[::2] = a >> 4
    out_dq = quant_range[out_dq]
    blocks = n // blocksize
    blocks += 1 if n % blocksize > 0 else 0
    rem = n % blocksize

    has_rem = rem > 0
    if has_rem:
        if out is None:
            out = torch.empty(quant_state.shape, dtype=quant_state.dtype, device=a.device)
        out_reshaped = out.reshape(-1)
        out_reshaped[: n - rem] = (
            out_dq[: n - rem].view(-1, blocksize) * absmax[: blocks - has_rem].view(-1, 1)
        ).reshape(-1)
        out_reshaped[n - rem :] = out_dq[n - rem :] * absmax[-1]
    else:
        out = (out_dq.view(-1, blocksize) * absmax.view(-1, 1)).reshape(out.shape).to(out.dtype)
    return out


def sum(a: torch.Tensor, b: torch.Tensor):
    return a + b


torch.manual_seed(0)


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


def mm4_ref(batch=1, seq=1, model=1024, hidden=1024):
    # TORCH_COMPILE_DEBUG = 1
    dequant_compiled = torch.compile(dequantize_nf4)

    # comp_sum = torch.compile(sum)

    di = torch.xpu

    quant_blocksize = 64
    a = torch.rand(hidden, model, device="xpu").half()
    # ref = comp_sum(a, a)
    # print("ref: ", ref[:16])
    # exit(0)
    qa, SA = F.quantize_4bit(a, blocksize=quant_blocksize, quant_type="nf4")

    quant_state_code = [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]

    B_dq = torch.empty_like(a, dtype=torch.float16)
    di.synchronize()

    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)

    cache_size = 256 * 1024 * 1024
    cache = torch.empty(int(cache_size // 4), dtype=torch.int, device="xpu")
    qa = qa.to(device="xpu")
    SA.code = SA.code.to(device="xpu")
    SA.absmax = SA.absmax.to(device="xpu")
    a = a.to(device="xpu")

    start_event.record()
    for _ in range(5):
        cache.zero_()
        B_dq = dequantize_nf4(qa, SA, B_dq, SA.code, SA.absmax, SA.blocksize)
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
        B_dq = dequantize_nf4(qa, SA, B_dq, SA.code, SA.absmax, SA.blocksize)
    # Benchmark
    for i in range(n_repeat):
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        B_dq = dequantize_nf4(qa, SA, B_dq, SA.code, SA.absmax, SA.blocksize)
        end_event[i].record()
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    times_med = _summarize_statistics(times, None, "median")
    # print("    dequantize_nf4 (default torch): ", estimate_ms, "ms")
    print("   med dequantize_nf4 (default torch): ", times_med, "ms")

    out_B = torch.empty_like(a, dtype=torch.float16)
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        dequant_nf4_fp16(qa, SA, SA.absmax, out_B, SA.blocksize, "nf4")
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
        dequant_nf4_fp16(qa, SA, SA.absmax, out_B, SA.blocksize, "nf4")
    # Benchmark
    for i in range(n_repeat):
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        dequant_nf4_fp16(qa, SA, SA.absmax, out_B, SA.blocksize, "nf4")
        end_event[i].record()
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    times_med = _summarize_statistics(times, None, "median")
    # print("        dequant_nf4_fp16 (triton): ", estimate_ms, "ms")
    print("        med dequant_nf4_fp16 (triton): ", times_med, "ms")

    # print("from_kernel: ", out_B.view(-1, quant_blocksize))
    # print("from_py: ", B_dq.view(-1, quant_blocksize))
    max_diff = out_B - B_dq
    assert torch.allclose(
        out_B, B_dq, atol=1e-2, rtol=0
    ), f"dequantized weight not close to original, max diff: {max_diff} First failed"

    B_dq_c = torch.empty_like(a, dtype=torch.float16)
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        B_dq_c = dequant_compiled(qa, SA, B_dq_c, SA.code, SA.absmax, SA.blocksize)
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
        B_dq_c = dequant_compiled(qa, SA, B_dq_c, SA.code, SA.absmax, SA.blocksize)
    # Benchmark
    for i in range(n_repeat):
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        B_dq_c = dequant_compiled(qa, SA, B_dq_c, SA.code, SA.absmax, SA.blocksize)
        end_event[i].record()
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    times_med = _summarize_statistics(times, None, "median")
    # print("                 dequant_compiled: ", estimate_ms, "ms")
    print(" med dequant_compiled (torch compile): ", times_med, "ms")

    out_B = torch.empty_like(a, dtype=torch.float16)
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        dequant2d_nf4_fp16(qa, SA, SA.absmax, out_B, SA.blocksize, "nf4")
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
        dequant2d_nf4_fp16(qa, SA, SA.absmax, out_B, SA.blocksize, "nf4")
    # Benchmark
    for i in range(n_repeat):
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        dequant2d_nf4_fp16(qa, SA, SA.absmax, out_B, SA.blocksize, "nf4")
        end_event[i].record()
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    times_med = _summarize_statistics(times, None, "median")
    # print("        dequant2d_nf4_fp16 (triton): ", estimate_ms, "ms")
    print("      med dequant2d_nf4_fp16 (triton): ", times_med, "ms")

    # max_diff = out_B - B_dq_c
    # assert torch.allclose(
    #     out_B, B_dq_c, atol=1e-2, rtol=0
    # ), f"dequantized weight not close to original, max diff: {max_diff} Second failed"


mm4_ref()
