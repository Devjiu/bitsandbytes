from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F_T

from bitsandbytes.utils import QuantState
import triton
import triton.language as tl

from .base import Backend
from .cpu_xpu_common import (
    dequantize_4bit_impl,
    double_quant_impl,
    gemm_4bit_impl,
    int8_linear_matmul_impl,
    int8_mm_dequant_impl,
    quantize_4bit_impl,
)

Tensor = torch.Tensor


# @triton.autotune(
#     configs=[
#         # triton.Config({'SPLIT_SIZE': 64}),
#         # triton.Config({'SPLIT_SIZE': 128}),
#         # triton.Config({'SPLIT_SIZE': 256}),
#         triton.Config({'SPLIT_SIZE': 512}),
#         # triton.Config({'SPLIT_SIZE': 1024}),
#         # triton.Config({'SPLIT_SIZE': 2048}),
#         # triton.Config({'SPLIT_SIZE': 4096}),
#         # triton.Config({'SPLIT_SIZE': 8192}),
#         # triton.Config({'SPLIT_SIZE': 16384}),
#     ],
#     key=['SPLIT_SIZE'],
# )
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


@triton.jit
def dequant_4bit_kernel_2d(
    a_ptr, c_ptr, quant_ptr, absmax_ptr, #
    R, C, #
    stride_a_row, stride_a_col, #
    QUANT_BLOCK: tl.constexpr,  #
    SPLIT_ROW: tl.constexpr,    #
    SPLIT_COL: tl.constexpr,    #
    GROUP_SIZE_M: tl.constexpr, #
):
    pid = tl.program_id(axis=0)
    # print("R ", R, " C ", C)
    num_pid_m = tl.cdiv(R, SPLIT_ROW)
    num_pid_n = tl.cdiv(C, SPLIT_COL)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_r = pid_m * SPLIT_ROW
    start_c = pid_n * SPLIT_COL

    offs_a_row = start_r + tl.arange(0, SPLIT_ROW)
    offs_a_row = tl.where(offs_a_row < R, offs_a_row, 0)
    offs_a_col = start_c + tl.arange(0, SPLIT_COL)
    offs_a_col = tl.where(offs_a_col < C, offs_a_col, 0)
    # offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_a_row = tl.max_contiguous(tl.multiple_of(offs_a_row, SPLIT_ROW), SPLIT_ROW)
    offs_a_col = tl.max_contiguous(tl.multiple_of(offs_a_col, SPLIT_COL), SPLIT_COL)

    # print("offs row: ", offs_a_row)
    # print("offs col: ", offs_a_col)
    # print("strides row: ", stride_a_row, " stride col: ", stride_a_col)
    # offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, SPLIT_COL), SPLIT_COL)
    # offs_k = tl.arange(0, BLOCK_SIZE_K)
    offsets = (offs_a_row[:, None] * stride_a_row + offs_a_col[None, :] * stride_a_col)
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
    offsets = (offs_a_row[:, None] * stride_a_row * 2 + offs_a_col[None, :] * stride_a_col)
    # print("out offsets: ", offsets)
    # offs = out_block_start + tl.arange(0, SPLIT_SIZE * 2)
    mask = offsets < num_paired_elements * 2
    # c_mask = (offs_a_row[:, None] < R) & (offs_a_col[None, :] < C * 2)
    # tl.store(c_ptr + offs, out_dq, mask)
    tl.store(c_ptr + offsets, out_dq, mask)

# @triton.autotune(
#     configs=[
#         triton.Config({'SPLIT_SIZE': 64}),
#         triton.Config({'SPLIT_SIZE': 128}),
#         triton.Config({'SPLIT_SIZE': 256}),
#         triton.Config({'SPLIT_SIZE': 512}),
#         triton.Config({'SPLIT_SIZE': 1024}),
#         triton.Config({'SPLIT_SIZE': 2048}),
#         triton.Config({'SPLIT_SIZE': 4096}),
#         triton.Config({'SPLIT_SIZE': 8192}),
#         triton.Config({'SPLIT_SIZE': 16384}),
#     ],
#     key=['SPLIT_SIZE'],
# )
@triton.jit
def dequant_8bit_kernel(
    a_ptr,
    c_ptr,
    quant_ptr,
    absmax_ptr,
    bias_ptr,
    num_paired_elements,
    QUANT_BLOCK: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * SPLIT_SIZE
    offsets = block_start + tl.arange(0, SPLIT_SIZE)
    mask = offsets < num_paired_elements

    a = tl.load(a_ptr + offsets, mask)
    a = a.to(tl.uint8, bitcast=True)

    # print("a: ", a)

    bias = tl.load(bias_ptr)
    # print("bias: ", bias)

    # apply conversion
    scaled_int8 = tl.load(quant_ptr + a, mask)

    # print("scaled: ", scaled_int8)

    abs_blocks_lim = (num_paired_elements // QUANT_BLOCK) * QUANT_BLOCK + num_paired_elements % QUANT_BLOCK
    abs_offsets = offsets // QUANT_BLOCK
    mask_blocked = offsets < abs_blocks_lim

    absmax = tl.load(absmax_ptr + abs_offsets, mask_blocked)
    # print("absmax: ", absmax)

    # apply scales
    out_dq = scaled_int8 * absmax
    # print("mul: ", out_dq)
    out_dq = out_dq + bias
    # print("biased: ", out_dq)

    # out_block_start = pid * SPLIT_SIZE
    offs = block_start + tl.arange(0, SPLIT_SIZE)
    mask = offs < num_paired_elements
    tl.store(c_ptr + offs, out_dq, mask)


def dequant_int8_fp16(
    A_nf4: torch.Tensor,
    bias: torch.Tensor,
    quant_state: QuantState,
    absmax: torch.Tensor,
    out: torch.Tensor,
    quant_blocksize: int = 64,
):
    # breakpoint()
    # print("absmax_orig: ", A_nf4)
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    quant_state_code = quant_state.code.to(device=DEVICE)

    number_of_paired_elements = A_nf4.numel()
    # we assume that split_size > quant_blocksize

    SPLIT_SIZE = 256
    # grid = lambda META: (triton.cdiv(number_of_paired_elements, META["SPLIT_SIZE"]), )
    grid = (triton.cdiv(number_of_paired_elements, SPLIT_SIZE),)
    # print("split: ", split_size, " grid: ", grid)
    # start = time.time()
    dequant_8bit_kernel[grid](
        A_nf4, out, quant_state_code, absmax, bias, number_of_paired_elements, quant_blocksize, SPLIT_SIZE
    )
    # print("out: ", out)
    return out


def dequant_8bit(A, offset, quant_state):
    assert A.dtype == torch.uint8
    # print("[ref] absmax A: ", A)
    # print("[ref] bias: ", offset)
    absmax = quant_state.code[A.reshape(-1).int()]
    # print("[ref] scaled A: ", absmax)
    blocks = absmax.shape[-1] // 256
    res = absmax.shape[-1] % 256
    if res != 0:
        absmax = F_T.pad(absmax, (0, 256 - res), mode="constant", value=0)
    absmax = (absmax.view(-1, 256) * quant_state.absmax.view(-1, 1)).to(quant_state.dtype).reshape(-1)
    absmax = absmax[: blocks * 256 + res]
    absmax = absmax.reshape(A.shape)
    absmax += offset
    return absmax

def dequantize_nf4(a: torch.Tensor, out: torch.Tensor, quant_range: torch.Tensor, absmax: torch.Tensor, blocksize):
    a = a.reshape(-1)
    out_dq = torch.empty(a.size(0) * 2, dtype=torch.int32)
    n = out_dq.numel()
    print("initial A: ", a)
    print("initial A: ", a.shape)
    # higher 4bits from uint8 packed tensor
    out_dq[1::2] = a & 0xF
    # lower 4bits
    out_dq[::2] = a >> 4
    out_dq = quant_range[out_dq]
    print("ref out_dq nf4: ", out_dq)
    blocks = n // blocksize
    blocks += 1 if n % blocksize > 0 else 0
    rem = n % blocksize

    has_rem = rem > 0
    if has_rem:
        assert False and "not implemented"
    else:
        print("out_dq reshaped: ", out_dq.view(-1, blocksize).shape)
        # print("absmax reshaped: ", absmax.view(-1, 1))
        print("absmax reshaped: ", absmax.view(-1, 1).shape)
        print("[dequantize_nf4] out shape: ", out.shape)
        print("ref mul: ", out_dq.view(-1, blocksize) * absmax.view(-1, 1))
        out = (out_dq.view(-1, blocksize) * absmax.view(-1, 1)).reshape(out.shape).to(out.dtype)
    return out


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
        # import time
        # start = time.time()
        # raise NotImplementedError("Fuck")
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
        # max_diff = absmax_ref - absmax

        # print("triton absmax_ref: ", absmax)
        # print("    python absmax: ", absmax_ref)
        # assert torch.allclose(
        #     absmax_ref, absmax, atol=1e-2, rtol=0
        # ), f"dequantized weight not close to original, max diff: {max_diff} First failed"
        # exit(0)
        # dequant_nf4_fp16(A, quant_state, absmax, out, blocksize, quant_type)
        # print("dequant_8bit: ", (time.time() - start), "s")

    if out is None:
        out = torch.empty(quant_state.shape, dtype=quant_state.dtype, device=A_nf4.device)

    # It's will be processed as an array, so
    # actual length is row * col
    # Elements are in uint8 format, so interleaved
    # so total amount of data is 2 * elem_count
    number_of_paired_elements = A_nf4.numel()
    # we assume that split_size > quant_blocksize

    import math
    SPLIT_R = 16
    SPLIT_C = 16
    GROUP_M = 2
    R, C = out.shape
    C = C // 2

    # orig_uint8 = A_nf4.to(torch.uint8)
    # print("uint8 shape: ", orig_uint8.shape, " stride: ", orig_uint8.stride())
    # shaped_A = orig_uint8.view(R, C)
    # print("orig: ", orig_uint8[128:132, :])
    # print("shaped: ", shaped_A.shape, " stride: ", shaped_A.stride())
    # # shaped_uint8 = shaped_A.to(torch.uint8)
    # print("ref A: ", shaped_A[:4, :4])
    # high = shaped_A[:4, :4] & 0xF
    # low  = shaped_A[:4, :4] >> 4
    # print("ref A high: ", high)
    # print("ref A  low: ", low)

    # qs_cpu = quant_state_code.cpu().half()
    # print("quant code: ", qs_cpu)
    # h_cpu = high.cpu().view(-1)
    # l_cpu = low.cpu().view(-1)
    # print("h_cpu ", h_cpu, " l_cpu ", l_cpu)
    # out_dq = torch.empty(16, dtype=torch.int32)
    # out_dq[:] = h_cpu
    # h_nf4 = qs_cpu[out_dq]
    # out_dq = torch.empty(16, dtype=torch.int32)
    # out_dq[:] = l_cpu
    # l_nf4 = qs_cpu[out_dq]
    # print("ref A high nf4: ", h_nf4.view(4, 4))
    # print("ref A  low nf4: ", l_nf4.view(4, 4))

    # dq_ref = dequantize_nf4(A_nf4.cpu(), out.cpu(), qs_cpu, absmax.cpu(), quant_state.blocksize)
    # print("[ref] dequantized: ", dq_ref[:8, :8])
    # R = int(math.sqrt(R*2))
    # C = R
    # grid = lambda META: (triton.cdiv(number_of_paired_elements, SPLIT_SIZE), )
    grid = (triton.cdiv(R, SPLIT_R) * triton.cdiv(C, SPLIT_C), )
    # print("split: ", split_size, " grid: ", grid)
    # start = time.time()
    # print("A_nf4 shape: ", A_nf4.shape, " stride: ", A_nf4.stride())
    # print("out shape: ", out.shape, " stride: ", out.stride())
    dequant_4bit_kernel_2d[grid](
        A_nf4, out, quant_state_code, absmax, R, C, C, out.stride(1), quant_state.blocksize, SPLIT_R, SPLIT_C, GROUP_M
    )

    if transpose:
        print("Transposing!")
        out = out.t()

    return out

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


def assert_on_xpu(tensors):
    on_xpu = True
    for t in tensors:
        if t is None:
            continue  # NULL pointers are fine
        on_xpu &= t.device.type == "xpu"
    if not on_xpu:
        raise TypeError(
            "All input tensors need to be on XPU, but found some tensors to not be on XPU:\n"
            f" {[(t.shape, t.device) if isinstance(t, Tensor) else None for t in tensors]}"
        )
    return on_xpu


class XPUBackend(Backend):
    mm_dequant_compute_dtype = torch.bfloat16
    mm_dequant_output_dtype = torch.bfloat16

    def int8_double_quant(
        self,
        A: torch.Tensor,
        col_stats: Optional[torch.Tensor] = None,
        row_stats: Optional[torch.Tensor] = None,
        out_col: Optional[torch.Tensor] = None,
        out_row: Optional[torch.Tensor] = None,
        threshold=0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert_on_xpu([A, col_stats, row_stats, out_col, out_row])
        output = double_quant_impl(A, col_stats, row_stats, out_col, out_row, threshold)
        return output

    def transform(
        self,
        A: torch.Tensor,
        to_order: str,
        from_order="row",
        out: Optional[torch.Tensor] = None,
        transpose=False,
        state: Optional[Tuple[torch.Size, str]] = None,
        ld=None,
    ):
        """
        Transform tensor A to to_order. It is originally designed for CUDA.
        For XPU, it returns the original tensor if transpose=False.
        Otherwise, it returns the transpose of A
        """
        assert_on_xpu([A, out])
        if transpose:
            if out is not None:
                out.copy_(A.T)
            else:
                out = A.T
        else:
            if out is not None:
                out.copy_(A)
            else:
                out = A
        return out, state

    def int8_linear_matmul(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        dtype=torch.int32,
    ) -> torch.Tensor:
        assert_on_xpu([A, B])
        output = int8_linear_matmul_impl(A, B, out, dtype)
        return output

    def int8_mm_dequant(
        self,
        A: torch.Tensor,
        row_stats: torch.Tensor,
        col_stats: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert_on_xpu([A, row_stats, col_stats, out, bias])
        output = int8_mm_dequant_impl(
            A,
            row_stats,
            col_stats,
            out,
            bias,
            self.mm_dequant_compute_dtype,
            self.mm_dequant_output_dtype,
        )
        return output

    def int8_vectorwise_dequant(self, A, stats):
        return super().int8_vectorwise_dequant(A, stats)

    def int8_vectorwise_quant(self, A: torch.Tensor, threshold=0.0):
        # TODO: We can optimize this as we don't actually need column-wise quant.
        out, _, stats, _, outlier_cols = self.int8_double_quant(A, threshold=threshold)
        return out, stats, outlier_cols

    def extract_outliers(
        self,
        A: torch.Tensor,
        SA: Tuple[torch.Size, str],
        idx: torch.Tensor,
    ) -> torch.Tensor:
        assert_on_xpu([A])
        output = A[:, idx].contiguous()
        return output

    def quantize_4bit(
        self,
        A: torch.Tensor,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize=64,
        compress_statistics=False,
        quant_type: Literal["fp4", "nf4"] = "fp4",
        quant_storage=torch.uint8,
    ) -> Tuple[torch.Tensor, QuantState]:
        if blocksize is None:
            blocksize = 64
        assert_on_xpu([A, absmax, out])
        output = quantize_4bit_impl(A, absmax, out, blocksize, compress_statistics, quant_type, quant_storage)
        return output

    def dequantize_4bit(
        self,
        A: torch.Tensor,
        quant_state: Optional[QuantState] = None,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize: int = 64,
        quant_type: Literal["fp4", "nf4"] = "fp4",
    ) -> torch.Tensor:
        if blocksize is None:
            blocksize = 64
        assert_on_xpu([A, absmax, out])
        if quant_type == "nf4":
            output = dequant_nf4_fp16(A, quant_state, absmax, out, blocksize, quant_type)
            return output

        if quant_type == "nf4" and getattr(quant_state, "ipex", False):
            output = torch.ops.torch_ipex.dequantize_4bit(A, "nf4", quant_state.shape, absmax, None, blocksize).t()
        else:
            output = dequantize_4bit_impl(A, quant_state, absmax, out, blocksize, quant_type)

        return output

    def gemv_4bit(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        transposed_A=False,
        transposed_B=False,
        state: QuantState = None,
    ) -> torch.Tensor:
        assert_on_xpu([A, B, out])
        if state is None:
            raise ValueError("state cannot be None. gemv_4bit() requires the state from quantize_4bit()")
        output = gemm_4bit_impl(A, B, out, transposed_A, transposed_B, state)
        return output

    def dequantize_blockwise(
        self,
        A: torch.Tensor,
        quant_state: Optional[QuantState] = None,
        absmax: Optional[torch.Tensor] = None,
        code: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize: int = 4096,
        nested=False,
    ) -> torch.Tensor:
        raise NotImplementedError

    def quantize_blockwise(
        self,
        A: torch.Tensor,
        code: Optional[torch.Tensor] = None,
        absmax: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        blocksize=4096,
        nested=False,
    ) -> Tuple[torch.Tensor, QuantState]:
        raise NotImplementedError

    def optimizer_update_8bit_blockwise(
        self,
        optimizer_name: str,
        g: torch.Tensor,
        p: torch.Tensor,
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
    ) -> None:
        raise NotImplementedError

    def optimizer_update_32bit(
        self,
        optimizer_name: str,
        g: torch.Tensor,
        p: torch.Tensor,
        state1: torch.Tensor,
        beta1: float,
        eps: float,
        step: int,
        lr: float,
        state2: Optional[torch.Tensor] = None,
        beta2: float = 0.0,
        beta3: float = 0.0,
        alpha: float = 0.0,
        weight_decay: float = 0.0,
        gnorm_scale: float = 1.0,
        unorm_vec: Optional[torch.Tensor] = None,
        max_unorm: float = 0.0,
        skip_zeros=False,
    ) -> None:
        raise NotImplementedError
