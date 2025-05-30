from collections.abc import Sequence

from typing import Any, Optional, Union

import torch

from . import triton_kernels

# currently codes unused, kept for reference
# Should be the same for quant/dequant
# from bitsandbytes.functional import get_4bit_type
# _FP4_QUANT_TABLE = get_4bit_type("fp4", device="xpu")
# _NF4_QUANT_TABLE = get_4bit_type("nf4", device="xpu")


import math
import statistics

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




def quantize_blockwise(A: torch.Tensor, code: torch.Tensor, blocksize: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)
    # torch._check(A.dtype == torch.float32, lambda: f"A must be float32 on xpu, got {A.dtype}")

    n = A.numel()
    blocks = -(n // -blocksize)

    absmax = torch.empty((blocks,), device=A.device, dtype=A.dtype)
    out = torch.empty_like(A.flatten(), dtype=torch.uint8)

    triton_kernels.quantize_blockwise_triton(A, blocksize, code, blocks, absmax, out)
    out = out.reshape(A.shape)

    return out, absmax.float()


def dequantize_blockwise(
    A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype
) -> torch.Tensor:
    torch._check_is_size(blocksize)
    torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")
    # torch._check(dtype == torch.float32, lambda: f"dtype must be float32 on xpu, got {dtype}")

    out = torch.empty_like(A, dtype=dtype, device=A.device)
    triton_kernels.dequant_int8_blockwise(
        A,
        code,
        absmax,
        out,
        blocksize,
    )

    return out


def dequantize_blockwise_inplace(
    A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype, out: torch.Tensor
) -> None:
    torch._check_is_size(blocksize)
    torch._check(A.dtype == torch.uint8, lambda: f"A must be uint8, got {A.dtype}")
    torch._check(out.shape == A.shape, lambda: f"Expected out.shape == {A.shape}, got {out.shape}")
    torch._check(out.device == A.device, lambda: f"Expected out.device == {A.device}, got {out.device}")
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")

    triton_kernels.dequant_int8_blockwise(
        A,
        code,
        absmax,
        out,
        blocksize,
    )


def quantize_4bit(
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check_is_size(blocksize)
    # torch._check(quant_type == "nf4", lambda: f"quant_type must be nf4 on CPU, got {quant_type}")
    torch._check(
        A.dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit quantization only supports 16/32-bit floats, but got {A.dtype}",
    )

    n = A.numel()

    # TODO: Support when weight matrix is not divisible by blocksize
    # torch._check(n % blocksize == 0, lambda: f"n must be divisible by blocksize, got {n} and {blocksize}")

    blocks = -(n // -(blocksize * 2))

    absmax = torch.empty((blocks * 2,), device=A.device, dtype=A.dtype)
    out = torch.empty((n // 2, 1), device=A.device, dtype=torch.uint8)

    triton_kernels.quantize_4bit_blockwise_triton(
        A, blocksize, quant_type, blocks, absmax, num_elements=n, quantized_out=out
    )
    packed = out

    if quant_storage != torch.uint8:
        packed = out.squeeze().view(quant_storage).unsqueeze(1)

    return packed, absmax.float()


def dequantize_4bit(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    torch._check_is_size(blocksize)
    # torch._check(quant_type == "nf4", lambda: f"quant_type must be nf4 on XPU, got {quant_type}")
    torch._check(
        dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit dequantization only supports 16/32-bit floats, but got {dtype}",
    )
    # torch._check(
    #     A.dtype == torch.uint8,
    #     lambda: f"Blockwise 4bit dequantization on XPU only supports uint8 storage, got {A.dtype}",
    # )
    # Check if this is fine and fast
    if A.dtype != torch.uint8:
        A = A.squeeze().view(torch.uint8).unsqueeze(1)

    out = torch.empty(shape, dtype=dtype, device=A.device)

    triton_kernels._dequantize_4bit_gather(A, absmax, blocksize, quant_type, dtype, out=out)

    # fused_sequantize_martmul = measure_gpu(triton_kernels._dequantize_4bit_impl, A, absmax, blocksize, quant_type, dtype, out)
    # print("  indirect load: ", fused_sequantize_martmul, " ms ")
    # fused_sequantize_martmul = measure_gpu(triton_kernels._dequantize_4bit_gather, A, absmax, blocksize, quant_type, dtype, out)
    # print("         gather: ", fused_sequantize_martmul, " ms ")

    return out


def dequantize_4bit_inplace(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    shape: Sequence[int],
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    torch._check(out.shape == shape, lambda: f"Expected out.shape == {shape}, got {out.shape}")
    torch._check(out.dtype == dtype, lambda: f"Expected out.dtype == {dtype}, got {out.dtype}")
    triton_kernels._dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out=out)


def gemv_4bit_separate_calls(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
) -> torch.Tensor:
    B_dq_triton = torch.empty(shapeB, dtype=A.dtype, device=A.device)

    triton_kernels._dequantize_4bit_impl_passing_code(
        B,
        absmax,
        blocksize,
        code,
        dtype=A.dtype,
        out=B_dq_triton,
    )
    return triton_kernels.triton_matmul(
        A,
        B_dq_triton.T,
    )

def gemv_4bit(
    A: torch.Tensor,
    B: torch.Tensor,
    shapeB: Sequence[int],
    absmax: torch.Tensor,
    code: torch.Tensor,
    blocksize: int,
) -> torch.Tensor:
    if B.dtype != torch.uint8:
        B = B.squeeze().view(torch.uint8).unsqueeze(1)

    fused_sequantize_martmul = measure_gpu(triton_kernels.matmul, A, B, shapeB, code, absmax, blocksize)
    print("             fused dequantization + matmul: ", fused_sequantize_martmul, " ms ")
    fused_sequantize_martmul = measure_gpu(gemv_4bit_separate_calls, A, B, shapeB, absmax, code, blocksize)
    print("  split 2 calls dequantization than matmul: ", fused_sequantize_martmul, " ms ")

    # return triton_kernels.matmul(A, B, shapeB, code=code, absmax=absmax, blocksize=blocksize)

    return gemv_4bit_separate_calls(
        A,
        B,
        shapeB,
        absmax,
        code,
        blocksize,
    )

    return torch.nn.functional.linear(
        A,
        B_dq_triton,
        bias=None,
    )


def adam_8bit_blockwise_grad(
        p: torch.Tensor,
        g: torch.Tensor,
        state1: torch.Tensor,
        state2: Optional[torch.Tensor],
        beta1: float,
        beta2: float,
        beta3: float,
        alpha: float,
        eps: float,
        step: int,
        lr: float,
        qmap1: torch.Tensor,
        qmap2: Optional[torch.Tensor],
        absmax1: torch.Tensor,
        absmax2: Optional[torch.Tensor],
        weight_decay: float = 0.0,
        gnorm_scale: float = 1.0,
        skip_zeros=False,
        n: int = 0,
):
    print("beta1: ", type(beta1), " beta2: ", type(beta2), " step: ", type(step))
    print("beta1: ", beta1, " beta2: ", beta2, " step: ", step)
    triton_kernels.optim_kernel_call(
        p,
        g,
        state1,
        state2,
        beta1,
        beta2,
        beta3,
        alpha,
        eps,
        step,
        lr,
        qmap1,
        qmap2,
        absmax1,
        absmax2,
        weight_decay=weight_decay,
        gnorm_scale=gnorm_scale,
        skip_zeros=skip_zeros,
        n=n
    )