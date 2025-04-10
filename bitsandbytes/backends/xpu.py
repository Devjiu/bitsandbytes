from typing import Literal, Optional, Tuple

import torch

from bitsandbytes.functional import pad as BBpad
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


@triton.jit
def dequant_kernel(
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


def dequant_8bit(A, offset, quant_state):
    assert A.dtype == torch.uint8
    absmax = quant_state.code[A.reshape(-1).int()]
    blocks = absmax.shape[-1] // 256
    res = absmax.shape[-1] % 256
    if res != 0:
        absmax = BBpad(absmax, (0, 256 - res), mode="constant", value=0)
    absmax = (absmax.view(-1, 256) * quant_state.absmax.view(-1, 1)).to(quant_state.dtype).reshape(-1)
    absmax = absmax[: blocks * 256 + res]
    absmax = absmax.reshape(A.shape)
    absmax += offset
    return absmax


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
        absmax = dequant_8bit(absmax, quant_state.offset, quant_state.state2)
        # print("dequant_8bit: ", (time.time() - start), "s")

    if out is None:
        out = torch.empty(quant_state.shape, dtype=quant_state.dtype, device=A_nf4.device)

    # It's will be processed as an array, so
    # actual length is row * col
    # Elements are in uint8 format, so interleaved
    # so total amount of data is 2 * elem_count
    number_of_paired_elements = A_nf4.numel()
    # we assume that split_size > quant_blocksize
    split_size = 2048

    grid = (number_of_paired_elements // split_size + 1,)
    # start = time.time()
    dequant_kernel[grid](
        A_nf4, out, quant_state_code, absmax, number_of_paired_elements, quant_state.blocksize, split_size
    )
    # print("dequant_kernel: ", (time.time() - start), "s")

    if transpose:
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
