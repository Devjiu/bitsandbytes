import torch

import triton
import triton.language as tl
# from bitsandbytes.functional import get_4bit_type
# _FP4_QUANT_TABLE = get_4bit_type("fp4", device="xpu")
# _NF4_QUANT_TABLE = get_4bit_type("nf4", device="xpu")

# @triton.autotune(
#     configs=[
#         # triton.Config({'SPLIT_SIZE': 64}),
#         # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
#         # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
#         # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
#         # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'auto'}, num_stages=4, num_warps=32),
#         # triton.Config({'SPLIT_SIZE': 128}),
#         # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
#         # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
#         # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
#         # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'auto'}, num_stages=4, num_warps=32),
#         triton.Config({"SPLIT_SIZE": 256}),
#         # triton.Config({'SPLIT_SIZE': 256, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
#         # triton.Config({'SPLIT_SIZE': 256, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
#         triton.Config({"SPLIT_SIZE": 512}),
#         # triton.Config({'SPLIT_SIZE': 1024}),
#     ],
#     key=["num_paired_elements", "QUANT_BLOCK"],
# )
@triton.jit
def dequant_8bit_kernel(
    a_ptr,
    c_ptr,
    quant_ptr,
    absmax_ptr,
    num_paired_elements,
    QUANT_BLOCK: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * SPLIT_SIZE
    offsets = block_start + tl.arange(0, SPLIT_SIZE)
    mask = offsets < num_paired_elements

    a = tl.load(a_ptr + offsets, mask)
    a = a.to(tl.uint8)

    # apply conversion
    scaled_int8 = tl.load(quant_ptr + a, mask)

    abs_blocks_lim = (num_paired_elements // QUANT_BLOCK) * QUANT_BLOCK + num_paired_elements % QUANT_BLOCK
    abs_offsets = offsets // QUANT_BLOCK
    mask_blocked = offsets < abs_blocks_lim

    absmax = tl.load(absmax_ptr + abs_offsets, mask_blocked)
    # apply scales
    out_dq = scaled_int8 * absmax

    offs = block_start + tl.arange(0, SPLIT_SIZE)
    mask = offs < num_paired_elements
    tl.store(c_ptr + offs, out_dq, mask)


def dequant_int8_blockwise(
    A_nf4: torch.Tensor,
    quant_state_code: torch.Tensor,
    absmax: torch.Tensor,
    out: torch.Tensor,
    quant_blocksize: int = 64,
):
    number_of_paired_elements = A_nf4.numel()

    SPLIT_SIZE = 256
    # grid = lambda META: (triton.cdiv(number_of_paired_elements, META["SPLIT_SIZE"]),)
    grid = (triton.cdiv(number_of_paired_elements, SPLIT_SIZE),)
    dequant_8bit_kernel[grid](
        A_nf4,
        out,
        quant_state_code,
        absmax,
        number_of_paired_elements,
        quant_blocksize,
        SPLIT_SIZE,
    )
    return out


# @triton.autotune(
#     configs=[
#         triton.Config({"SPLIT_NUM_BLOCKS": 1, "grf_mode": "auto"}, num_stages=4, num_warps=32),
#         triton.Config({"SPLIT_NUM_BLOCKS": 2, "grf_mode": "auto"}, num_stages=4, num_warps=32),
#         triton.Config({"SPLIT_NUM_BLOCKS": 1}),
#         triton.Config({"SPLIT_NUM_BLOCKS": 2}),
#     ],
#     key=["n_elements"],
# )
@triton.jit
def quantize_blockwise_kernel(
    A_ptr,
    code_ptr,
    absmax_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    CODE_SIZE: tl.constexpr,
    SPLIT_NUM_BLOCKS: tl.constexpr,
):
    block_start_idx = tl.program_id(0) * SPLIT_NUM_BLOCKS
    thread_idx = tl.arange(0, SPLIT_NUM_BLOCKS * BLOCK_SIZE)

    offsets = block_start_idx * BLOCK_SIZE + thread_idx
    mask = offsets < n_elements

    A = tl.load(A_ptr + offsets, mask=mask, other=0.0)

    # To be able process several blocks -> (BLOCK_SIZE, SPLIT_NUM_BLOCKS)
    A_reshaped = tl.reshape(A, (SPLIT_NUM_BLOCKS, BLOCK_SIZE))

    # Calculating absamax for each block
    absmax = tl.max(tl.abs(A_reshaped), axis=1)
    tl.store(absmax_ptr + block_start_idx + tl.arange(0, SPLIT_NUM_BLOCKS), absmax)

    A_normalized = A_reshaped / absmax[:, None]
    A_normalized = tl.clamp(A_normalized, -1.0, 1.0)

    lower_pivot = tl.zeros((SPLIT_NUM_BLOCKS, BLOCK_SIZE), dtype=tl.int32)
    upper_pivot = tl.full((SPLIT_NUM_BLOCKS, BLOCK_SIZE), CODE_SIZE - 1, dtype=tl.int32)

    for _ in range(8):  # ceil(log2(code_size)) = 8, actually, in general case should be input parameter
        pivot = (lower_pivot + upper_pivot) // 2
        val = tl.load(code_ptr + pivot)
        is_higher = A_normalized > val  # code[pivot]
        lower_pivot = tl.where(is_higher, pivot, lower_pivot)
        upper_pivot = tl.where(is_higher, upper_pivot, pivot)

    # Choose closest level
    lower_val = tl.load(code_ptr + lower_pivot)
    upper_val = tl.load(code_ptr + upper_pivot)
    lower_dist = tl.abs(A_normalized - lower_val)
    upper_dist = tl.abs(A_normalized - upper_val)
    quantized = tl.where(lower_dist <= upper_dist, lower_pivot, upper_pivot).to(tl.uint8)

    # too slow approach
    # diff = tl.abs(A_normalized[:, :, None] - code[None, None, :])
    # quantized = tl.argmin(diff, axis=2).to(tl.uint8)

    quantized_flat = tl.reshape(quantized, (BLOCK_SIZE * SPLIT_NUM_BLOCKS,))
    tl.store(out_ptr + offsets, quantized_flat, mask=mask)


def quantize_blockwise_triton(A, blocksize, code, blocks, absmax, quantized_out):
    n = A.numel()

    split_num_blocks = 1
    grid = (triton.cdiv(blocks, split_num_blocks),)
    # grid = lambda META: (triton.cdiv(blocks, META["SPLIT_NUM_BLOCKS"]),)
    quantize_blockwise_kernel[grid](
        A_ptr=A,
        code_ptr=code,
        absmax_ptr=absmax,
        out_ptr=quantized_out,
        n_elements=n,
        BLOCK_SIZE=blocksize,
        CODE_SIZE=code.numel(),
        SPLIT_NUM_BLOCKS=split_num_blocks,
    )

    return quantized_out, absmax


# Triton implementation of similar CUDA kernel to avoid loading code from csrc/kernels.cu::dQuantizeFP4
# @triton.autotune(
#     configs=[
#         triton.Config({"SPLIT_NUM_BLOCKS": 1, "grf_mode": "auto"}, num_stages=4, num_warps=32),
#         triton.Config({"SPLIT_NUM_BLOCKS": 2, "grf_mode": "auto"}, num_stages=4, num_warps=32),
#         triton.Config({"SPLIT_NUM_BLOCKS": 1}),
#         triton.Config({"SPLIT_NUM_BLOCKS": 2}),
#         triton.Config({"SPLIT_NUM_BLOCKS": 4}),
#         triton.Config({"SPLIT_NUM_BLOCKS": 8}),
#     ],
#     key=["n_elements"],
# )
@triton.jit
def quantize_fp4_blockwise_kernel(
    A_ptr,
    absmax_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    SPLIT_NUM_BLOCKS: tl.constexpr,
):
    PAIRED_SPLIT_NUM_BLOCKS: tl.constexpr = SPLIT_NUM_BLOCKS * 2
    block_start_idx = tl.program_id(0) * PAIRED_SPLIT_NUM_BLOCKS
    thread_idx = tl.arange(0, PAIRED_SPLIT_NUM_BLOCKS * BLOCK_SIZE)

    offsets = block_start_idx * BLOCK_SIZE + thread_idx
    mask = offsets < n_elements

    A = tl.load(A_ptr + offsets, mask=mask, other=0.0)

    # To be able process several blocks -> (PAIRED_SPLIT_NUM_BLOCKS, BLOCK_SIZE)
    A_reshaped = tl.reshape(A, (PAIRED_SPLIT_NUM_BLOCKS, BLOCK_SIZE))

    # Calculating absamax for each block
    absmax = tl.max(tl.abs(A_reshaped), axis=1)
    tl.store(absmax_ptr + block_start_idx + tl.arange(0, PAIRED_SPLIT_NUM_BLOCKS), absmax)

    A_normalized = A_reshaped / absmax[:, None]
    A_normalized = tl.clamp(A_normalized, -1.0, 1.0)

    sign = tl.where(A_normalized < 0, 0b1000, 0b0000)
    A_absf = tl.abs(A_normalized)

    result = tl.where(
        A_absf > 0.29166667,
        tl.where(
            A_absf > 0.583333, tl.where(A_absf > 0.8333333, 0b011, 0b010), tl.where(A_absf > 0.4166667, 0b101, 0b100)
        ),
        tl.where(
            A_absf > 0.0859375,
            tl.where(A_absf > 0.20833333, 0b0111, 0b0110),
            tl.where(A_absf > 0.00260417, 0b0001, 0b0000),
        ),
    )
    quantized = (result ^ sign).to(tl.uint8)

    quantized = quantized.reshape((PAIRED_SPLIT_NUM_BLOCKS, BLOCK_SIZE // 2, 2))
    left, right = quantized.split()
    packed = left << 4 | (right & 0xF)

    packed_flat = tl.reshape(packed, (BLOCK_SIZE * SPLIT_NUM_BLOCKS,))
    out_offsets = block_start_idx * BLOCK_SIZE // 2 + tl.arange(0, SPLIT_NUM_BLOCKS * BLOCK_SIZE)
    out_mask = out_offsets < n_elements // 2
    tl.store(out_ptr + out_offsets, packed_flat, mask=out_mask)


# Triton implementation of similar CUDA kernel to avoid loading code from csrc/kernels.cu::dQuantizeNF4
# @triton.autotune(
#     configs=[
#         triton.Config({"SPLIT_NUM_BLOCKS": 1, "grf_mode": "auto"}, num_stages=4, num_warps=32),
#         triton.Config({"SPLIT_NUM_BLOCKS": 2, "grf_mode": "auto"}, num_stages=4, num_warps=32),
#         triton.Config({"SPLIT_NUM_BLOCKS": 1}),
#         triton.Config({"SPLIT_NUM_BLOCKS": 2}),
#         triton.Config({"SPLIT_NUM_BLOCKS": 4}),
#         triton.Config({"SPLIT_NUM_BLOCKS": 8}),
#     ],
#     key=["n_elements"],
# )
@triton.jit
def quantize_nf4_blockwise_kernel(
    A_ptr,
    absmax_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    SPLIT_NUM_BLOCKS: tl.constexpr,
):
    PAIRED_SPLIT_NUM_BLOCKS: tl.constexpr = SPLIT_NUM_BLOCKS * 2
    block_start_idx = tl.program_id(0) * PAIRED_SPLIT_NUM_BLOCKS
    thread_idx = tl.arange(0, PAIRED_SPLIT_NUM_BLOCKS * BLOCK_SIZE)

    offsets = block_start_idx * BLOCK_SIZE + thread_idx
    mask = offsets < n_elements

    A = tl.load(A_ptr + offsets, mask=mask, other=0.0)

    # To be able process several blocks -> (PAIRED_SPLIT_NUM_BLOCKS, BLOCK_SIZE)
    A_reshaped = tl.reshape(A, (PAIRED_SPLIT_NUM_BLOCKS, BLOCK_SIZE))

    # Calculating absamax for each block
    absmax = tl.max(tl.abs(A_reshaped), axis=1)
    tl.store(absmax_ptr + block_start_idx + tl.arange(0, PAIRED_SPLIT_NUM_BLOCKS), absmax)

    A_normalized = A_reshaped / absmax[:, None]
    A_normalized = tl.clamp(A_normalized, -1.0, 1.0)

    result = tl.where(
        A_normalized > 0.03979014977812767,
        tl.where(
            A_normalized > 0.3893125355243683,
            tl.where(
                A_normalized > 0.6427869200706482,
                tl.where(A_normalized > 0.8614784181118011, 0b1111, 0b1110),
                tl.where(A_normalized > 0.5016634166240692, 0b1101, 0b1100),
            ),
            tl.where(
                A_normalized > 0.2035212516784668,
                tl.where(A_normalized > 0.2920137718319893, 0b1011, 0b1010),
                tl.where(A_normalized > 0.1202552504837513, 0b1001, 0b1000),
            ),
        ),
        tl.where(
            A_normalized > -0.33967943489551544,
            tl.where(
                A_normalized > -0.13791173323988914,
                tl.where(A_normalized > -0.045525018125772476, 0b0111, 0b0110),
                tl.where(A_normalized > -0.23460740596055984, 0b0101, 0b0100),
            ),
            tl.where(
                A_normalized > -0.6106329262256622,
                tl.where(A_normalized > -0.4599952697753906, 0b0011, 0b0010),
                tl.where(A_normalized > -0.8480964004993439, 0b0001, 0b0000),
            ),
        ),
    )
    quantized = result.to(tl.uint8)

    quantized = quantized.reshape((PAIRED_SPLIT_NUM_BLOCKS, BLOCK_SIZE // 2, 2))

    left, right = quantized.split()
    packed = left << 4 | (right & 0xF)

    packed_flat = tl.reshape(packed, (BLOCK_SIZE * SPLIT_NUM_BLOCKS,))
    out_offsets = block_start_idx * BLOCK_SIZE // 2 + tl.arange(0, SPLIT_NUM_BLOCKS * BLOCK_SIZE)
    out_mask = out_offsets < n_elements // 2
    tl.store(out_ptr + out_offsets, packed_flat, mask=out_mask)


def quantize_4bit_blockwise_triton(A, blocksize, quant_type, blocks, absmax, num_elements, quantized_out):
    # grid = lambda META: (triton.cdiv(blocks, META["SPLIT_NUM_BLOCKS"]),)
    split_num_blocks = 4
    grid = (triton.cdiv(blocks, split_num_blocks),)
    if quant_type == "fp4":
        quantize_fp4_blockwise_kernel[grid](
            A_ptr=A,
            absmax_ptr=absmax,
            out_ptr=quantized_out,
            n_elements=num_elements,
            BLOCK_SIZE=blocksize,
            SPLIT_NUM_BLOCKS=split_num_blocks,
        )
    else:
        quantize_nf4_blockwise_kernel[grid](
            A_ptr=A,
            absmax_ptr=absmax,
            out_ptr=quantized_out,
            n_elements=num_elements,
            BLOCK_SIZE=blocksize,
            SPLIT_NUM_BLOCKS=split_num_blocks,
        )
    return quantized_out, absmax


@triton.jit
def dequant_4bit_body_util(a, offsets, quant_ptr, absmax_ptr, n_elems, QUANT_BLOCK: tl.constexpr):
    PAIRED_QUANT_BLOCK: tl.constexpr = QUANT_BLOCK // 2
    mask = offsets < n_elems
    higher = a & 0xF
    # lower 4bits
    lower = a >> 4

    abs_offsets = offsets // PAIRED_QUANT_BLOCK
    absmax = tl.load(absmax_ptr + abs_offsets, mask=mask, other=1.0, eviction_policy="evict_last")

    # apply conversion
    lower_4 = tl.load(quant_ptr + lower, eviction_policy="evict_last")
    higher_4 = tl.load(quant_ptr + higher, eviction_policy="evict_last")

    mul_high = higher_4 * absmax
    mul_low = lower_4 * absmax
    out_dq = tl.interleave(mul_low, mul_high)
    return out_dq


@triton.jit
def dequant_4bit_body_util_gather(a, offsets, code, absmax_ptr, n_elems, SPLIT_SIZE: tl.constexpr, QUANT_BLOCK: tl.constexpr):
    PAIRED_QUANT_BLOCK: tl.constexpr = QUANT_BLOCK // 2
    mask = offsets < n_elems
    higher = a & 0xF
    # lower 4bits
    lower = a >> 4

    abs_offsets = offsets // PAIRED_QUANT_BLOCK
    absmax = tl.load(absmax_ptr + abs_offsets, mask=mask, other=1.0, eviction_policy="evict_last")

    # code shape num subgroups x code_size
    called_lower = tl.gather(code, lower, axis=1)
    called_higher = tl.gather(code, higher, axis=1)
    # lower_4 = tl.load(quant_ptr + lower, eviction_policy="evict_last")
    # higher_4 = tl.load(quant_ptr + higher, eviction_policy="evict_last")
    # lower_4 = called_lower.reshape(SPLIT_SIZE)
    # higher_4 = called_higher.reshape(SPLIT_SIZE)

    mul_low = called_lower * absmax
    mul_high = called_higher * absmax
    out_dq = tl.interleave(mul_low, mul_high)
    return out_dq

# Triton implementation of similar CUDA kernel to avoid loading code from csrc/kernels.cu::dDequantizeFP4Tree
@triton.jit
def dequantize_fp4_tree(val, absmax):
    # val: tl.tensor (uint8)
    # absmax: tl.tensor (float32/float16)
    #  00001100  00001011  00001001  00001111
    sign = tl.where((val & 0b1000) == 0b1000, -1.0, 1.0)  # -1
    third_bit = (val & 0b0100) == 0b0100  # True
    second_bit = (val & 0b0010) == 0b0010  # False
    first_bit = (val & 0b0001) == 0b0001  # False

    branch1 = tl.where(
        second_bit,
        tl.where(first_bit, 0.25, 0.16666667),  # 1111, 1110
        tl.where(first_bit, 0.5, 0.33333333),  # 1101, 1100
    )
    branch2 = tl.where(
        second_bit,
        tl.where(first_bit, 1.0, 0.66666667),  # 1011, 1010
        tl.where(first_bit, 0.00520833, 0.0),  # 1001, 1000
    )
    out = tl.where(third_bit, branch1, branch2)
    return out * sign * absmax


@triton.jit
def dequant_fp4_body_util(a, offsets, absmax_ptr, n_elems, QUANT_BLOCK: tl.constexpr):
    PAIRED_QUANT_BLOCK: tl.constexpr = QUANT_BLOCK // 2
    mask = offsets < n_elems
    higher = a & 0xF
    lower = a >> 4

    abs_offsets = offsets // PAIRED_QUANT_BLOCK
    absmax = tl.load(absmax_ptr + abs_offsets, mask=mask, other=1.0, eviction_policy="evict_last")
    mul_high = dequantize_fp4_tree(higher, absmax)
    mul_low = dequantize_fp4_tree(lower, absmax)
    out_dq = tl.interleave(mul_low, mul_high)
    return out_dq


# Triton implementation of similar CUDA kernel to avoid loading code from csrc/kernels.cu::dDequantizeNF4
@triton.jit
def dequantize_nf4_tree(val):
    # val: tl.tensor (uint8)
    cond0 = (val & 0b1000) == 0b1000
    cond1 = (val & 0b0100) == 0b0100
    cond2 = (val & 0b0010) == 0b0010
    cond3 = (val & 0b0001) == 0b0001

    # Positive branch (val & 0b1000) == 8
    branch_pos = tl.where(
        cond1,
        tl.where(
            cond2,
            tl.where(cond3, 1.0, 0.7229568362236023),  # 1111, 1110
            tl.where(cond3, 0.5626170039176941, 0.44070982933044434),  # 1101, 1100
        ),
        tl.where(
            cond2,
            tl.where(cond3, 0.33791524171829224, 0.24611230194568634),  # 1011, 1010
            tl.where(cond3, 0.16093020141124725, 0.07958029955625534),  # 1001, 1000
        ),
    )

    # Negative branch (val & 0b1000) == 0
    branch_neg = tl.where(
        cond1,
        tl.where(
            cond2,
            tl.where(cond3, 0.0, -0.09105003625154495),  # 0111, 0110
            tl.where(cond3, -0.18477343022823334, -0.28444138169288635),  # 0101, 0100
        ),
        tl.where(
            cond2,
            tl.where(cond3, -0.39491748809814453, -0.5250730514526367),  # 0011, 0010
            tl.where(cond3, -0.6961928009986877, -1.0),  # 0001, 0000
        ),
    )
    return tl.where(cond0, branch_pos, branch_neg)


@triton.jit
def dequant_nf4_body_util(a, offsets, absmax_ptr, n_elems, QUANT_BLOCK: tl.constexpr):
    PAIRED_QUANT_BLOCK: tl.constexpr = QUANT_BLOCK // 2
    mask = offsets < n_elems
    higher = a & 0xF
    # lower 4bits
    lower = a >> 4

    abs_offsets = offsets // PAIRED_QUANT_BLOCK
    absmax = tl.load(absmax_ptr + abs_offsets, mask=mask, other=1.0, eviction_policy="evict_last")
    mul_high = dequantize_nf4_tree(higher) * absmax
    mul_low = dequantize_nf4_tree(lower) * absmax
    out_dq = tl.interleave(mul_low, mul_high)
    return out_dq


# All such kernels are similar, so maybe code can be generalised.
# @triton.autotune(
#     configs=[
#         triton.Config({'SPLIT_SIZE': 64}),
# #         # # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
# #         # # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
# #         # # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
# #         # # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'auto'}, num_stages=4, num_warps=32),
#         triton.Config({'SPLIT_SIZE': 128}),
#         triton.Config({'SPLIT_SIZE': 128}, num_warps = 32, num_stages = 2),
# #         # triton.Config({'SPLIT_SIZE': 128}, num_warps = 4, num_stages = 4),
# #         # # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
# #         # # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
# #         # # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
# #         # # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'auto'}, num_stages=4, num_warps=32),
#         triton.Config({'SPLIT_SIZE': 256}),
#         triton.Config({'SPLIT_SIZE': 256}, num_warps = 32, num_stages = 2),
#         # triton.Config({'SPLIT_SIZE': 256}, num_warps = 4, num_stages = 4),
#         triton.Config({'SPLIT_SIZE': 512}),
#         triton.Config({'SPLIT_SIZE': 512}, num_warps = 32, num_stages = 2),
#         # triton.Config({'SPLIT_SIZE': 512}, num_warps = 4, num_stages = 4),
# #         # # triton.Config({'SPLIT_SIZE': 512, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
# #         # # triton.Config({'SPLIT_SIZE': 512, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
# #         # # triton.Config({'SPLIT_SIZE': 512, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
# #         # # triton.Config({'SPLIT_SIZE': 512, 'grf_mode': 'auto'}, num_stages=4, num_warps=32),
# #         # triton.Config({'SPLIT_SIZE': 1024}),
# #         # # triton.Config({'SPLIT_SIZE': 2048}),
# #         # # triton.Config({'SPLIT_SIZE': 4096}),
# #         # # triton.Config({'SPLIT_SIZE': 8192}),
# #         # # triton.Config({'SPLIT_SIZE': 16384}),
#     ],
#     key=['num_paired_elements'],
# )
@triton.jit
def dequant_4bit_kernel(
    a_ptr, c_ptr, quant_ptr, absmax_ptr, num_paired_elements, QUANT_BLOCK: tl.constexpr, SPLIT_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * SPLIT_SIZE
    offsets = block_start + tl.arange(0, SPLIT_SIZE)
    mask = offsets < num_paired_elements

    a = tl.load(a_ptr + offsets, mask, eviction_policy="evict_first")

    out_dq = dequant_4bit_body_util(
        a=a,
        offsets=offsets,
        quant_ptr=quant_ptr,
        absmax_ptr=absmax_ptr,
        n_elems=num_paired_elements,
        QUANT_BLOCK=QUANT_BLOCK,
    )

    out_block_start = pid * SPLIT_SIZE * 2
    offs = out_block_start + tl.arange(0, SPLIT_SIZE * 2)
    mask = offs < num_paired_elements * 2
    tl.store(c_ptr + offs, out_dq, mask)

# @triton.autotune(
#     configs=[
#         triton.Config({'SPLIT_SIZE': 64}),
#         triton.Config({'SPLIT_SIZE': 128}),
#         triton.Config({'SPLIT_SIZE': 128}, num_warps = 32, num_stages = 2),
#         triton.Config({'SPLIT_SIZE': 256}),
#         triton.Config({'SPLIT_SIZE': 256}, num_warps = 32, num_stages = 2),
#         triton.Config({'SPLIT_SIZE': 512}),
#         triton.Config({'SPLIT_SIZE': 512}, num_warps = 32, num_stages = 2),
#     ],
#     key=['num_paired_elements'],
# )
@triton.jit
def dequant_4bit_kernel_gather(
    a_ptr, c_ptr, quant_ptr, absmax_ptr, num_paired_elements, QUANT_BLOCK: tl.constexpr, SPLIT_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * SPLIT_SIZE
    subgroup_size: tl.constexpr = 16
    num_groups: tl.constexpr = SPLIT_SIZE // subgroup_size

    offsets = block_start + tl.arange(0, SPLIT_SIZE)
    mask = offsets < num_paired_elements

    # 2D offsets: shape [num_groups, 32]
    group_idx = tl.arange(0, num_groups)[:, None]      # shape [num_groups, 1]
    elem_idx = tl.arange(0, subgroup_size)[None, :]               # shape [1, subgroup_size]
    offsets_2d = block_start + group_idx * subgroup_size + elem_idx  # shape [num_groups, subgroup_size]
    # print("offsets_2d: ", offsets_2d.shape, " ", offsets_2d)
    mask = offsets_2d < num_paired_elements

    a = tl.load(a_ptr + offsets_2d, mask, eviction_policy="evict_first")
    # a = a.reshape(num_groups, subgroup_size)
    subgroup_id = tl.arange(0, num_groups)[:, None] * 0  # shape [num_groups, 1]
    code = tl.load(quant_ptr + subgroup_id + tl.arange(0, 16)[None, :])

    out_dq = dequant_4bit_body_util_gather(
        a=a,
        offsets=offsets_2d,
        code=code,
        absmax_ptr=absmax_ptr,
        n_elems=num_paired_elements,
        SPLIT_SIZE=SPLIT_SIZE,
        QUANT_BLOCK=QUANT_BLOCK,
    )

    out_dq = out_dq.reshape(SPLIT_SIZE * 2)

    out_block_start = pid * SPLIT_SIZE * 2
    offs = out_block_start + tl.arange(0, SPLIT_SIZE * 2)
    mask = offs < num_paired_elements * 2
    tl.store(c_ptr + offs, out_dq, mask)


# @triton.autotune(
#     configs=[
#         triton.Config({'SPLIT_SIZE': 128}, num_warps = 32, num_stages = 2),
#         triton.Config({'SPLIT_SIZE': 256}),
#         triton.Config({'SPLIT_SIZE': 256}, num_warps = 32, num_stages = 2),
#         triton.Config({'SPLIT_SIZE': 512}),
#         triton.Config({'SPLIT_SIZE': 512}, num_warps = 32, num_stages = 2),
#         triton.Config({'SPLIT_SIZE': 1024}, num_warps = 32, num_stages = 2),
#     ],
#     key=['num_paired_elements'],
# )
@triton.jit
def dequant_fp4_kernel(
    a_ptr, c_ptr, absmax_ptr, num_paired_elements, QUANT_BLOCK: tl.constexpr, SPLIT_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * SPLIT_SIZE
    offsets = block_start + tl.arange(0, SPLIT_SIZE)
    mask = offsets < num_paired_elements

    a = tl.load(a_ptr + offsets, mask, eviction_policy="evict_first")

    out_dq = dequant_fp4_body_util(
        a=a,
        offsets=offsets,
        absmax_ptr=absmax_ptr,
        n_elems=num_paired_elements,
        QUANT_BLOCK=QUANT_BLOCK,
    )

    out_block_start = pid * SPLIT_SIZE * 2
    offs = out_block_start + tl.arange(0, SPLIT_SIZE * 2)
    mask = offs < num_paired_elements * 2
    tl.store(c_ptr + offs, out_dq, mask)


# @triton.autotune(
#     configs=[
#         triton.Config({'SPLIT_SIZE': 128}, num_warps = 32, num_stages = 2),
#         triton.Config({'SPLIT_SIZE': 256}),
#         triton.Config({'SPLIT_SIZE': 256}, num_warps = 32, num_stages = 2),
#         triton.Config({'SPLIT_SIZE': 512}),
#         triton.Config({'SPLIT_SIZE': 512}, num_warps = 32, num_stages = 2),
#         triton.Config({'SPLIT_SIZE': 1024}, num_warps = 32, num_stages = 2),
#     ],
#     key=['num_paired_elements'],
# )
@triton.jit
def dequant_nf4_kernel(
    a_ptr, c_ptr, absmax_ptr, num_paired_elements, QUANT_BLOCK: tl.constexpr, SPLIT_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * SPLIT_SIZE
    offsets = block_start + tl.arange(0, SPLIT_SIZE)
    mask = offsets < num_paired_elements

    a = tl.load(a_ptr + offsets, mask, eviction_policy="evict_first")

    out_dq = dequant_nf4_body_util(
        a=a,
        offsets=offsets,
        absmax_ptr=absmax_ptr,
        n_elems=num_paired_elements,
        QUANT_BLOCK=QUANT_BLOCK,
    )

    out_block_start = pid * SPLIT_SIZE * 2
    offs = out_block_start + tl.arange(0, SPLIT_SIZE * 2)
    mask = offs < num_paired_elements * 2
    tl.store(c_ptr + offs, out_dq, mask)


def _dequantize_4bit_impl(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    # It's will be processed as an array, so
    # actual length is row * col
    # Elements are in uint8 format, so interleaved
    # so total amount of data is 2 * elem_count
    number_of_paired_elements = A.numel()
    # we assume that split_size > quant_blocksize

    SPLIT_SIZE = 256
    # grid = lambda META: (triton.cdiv(number_of_paired_elements, META['SPLIT_SIZE']), )
    grid = (triton.cdiv(number_of_paired_elements, SPLIT_SIZE),)
    if quant_type == "fp4":
        # dequant_fp4_kernel[grid](A, out, absmax, number_of_paired_elements, blocksize, SPLIT_SIZE)
        dequant_4bit_kernel[grid](A, out, _FP4_QUANT_TABLE, absmax, number_of_paired_elements, blocksize, SPLIT_SIZE)
    else:
        # dequant_nf4_kernel[grid](A, out, absmax, number_of_paired_elements, blocksize, SPLIT_SIZE)
        dequant_4bit_kernel[grid](A, out, _NF4_QUANT_TABLE, absmax, number_of_paired_elements, blocksize, SPLIT_SIZE)


def _dequantize_4bit_gather(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: str,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    # It's will be processed as an array, so
    # actual length is row * col
    # Elements are in uint8 format, so interleaved
    # so total amount of data is 2 * elem_count
    number_of_paired_elements = A.numel()
    # we assume that split_size > quant_blocksize

    SPLIT_SIZE = 128
    # grid = lambda META: (triton.cdiv(number_of_paired_elements, META['SPLIT_SIZE']), )
    grid = (triton.cdiv(number_of_paired_elements, SPLIT_SIZE),)
    if quant_type == "fp4":
        # dequant_fp4_kernel[grid](A, out, absmax, number_of_paired_elements, blocksize, SPLIT_SIZE)
        dequant_4bit_kernel_gather[grid](A, out, _FP4_QUANT_TABLE, absmax, number_of_paired_elements, blocksize, SPLIT_SIZE)
    else:
        # dequant_nf4_kernel[grid](A, out, absmax, number_of_paired_elements, blocksize, SPLIT_SIZE)
        dequant_4bit_kernel_gather[grid](A, out, _NF4_QUANT_TABLE, absmax, number_of_paired_elements, blocksize, SPLIT_SIZE)


def _dequantize_4bit_impl_passing_code(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    code: torch.Tensor,
    dtype: torch.dtype,
    out: torch.Tensor,
) -> None:
    number_of_paired_elements = A.numel()
    # we assume that split_size > quant_blocksize

    SPLIT_SIZE = 256
    # grid = lambda META: (triton.cdiv(number_of_paired_elements, META['SPLIT_SIZE']), )
    grid = (triton.cdiv(number_of_paired_elements, SPLIT_SIZE),)
    dequant_4bit_kernel[grid](A, out, code, absmax, number_of_paired_elements, blocksize, SPLIT_SIZE)


############## Experimental implementation fused Matmul with Dequantization ##############

# ==================================================


@triton.autotune(
    configs=[
        # # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=2,
        # #               num_warps=32),
        # # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4}, num_stages=2,
        # #               num_warps=32),
        # # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4}, num_stages=2,
        # #               num_warps=32),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4}),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4}),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 4}),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4}, num_stages=2,
        #               num_warps=32),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 4},
            num_stages=2,
            num_warps=32,
        ),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 4}, num_stages=2,
        #               num_warps=32),
        # # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=2,
        # #               num_warps=32),
        # # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=3,
        # #               num_warps=32),
        # # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=2,
        # #               num_warps=32),
        # # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=2,
        # #               num_warps=32),
    ],
    key=["M", "N", "K"],
)
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
    stride_az,
    stride_am,
    stride_ak,
    stride_bz,
    stride_bn,
    stride_bk,
    stride_cz,
    stride_cm,
    stride_cn,
    quant_ptr,
    absmax_ptr,
    num_paired_elements,
    ACCUMULATOR_DTYPE: tl.constexpr,
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
    bid = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offset_a = bid.to(tl.int64) * stride_az
    offset_b = bid.to(tl.int64) * stride_bz
    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_bk = tl.arange(0, BLOCK_SIZE_K // 2)

    b_offsets = offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr + offset_a,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr + offset_b,
        shape=(N, K // 2),
        strides=(stride_bn, stride_bk),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K // 2),
        order=(1, 0),
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUMULATOR_DTYPE)
    # dq_b = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=a_ptr.type.element_ty)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a_blck = tl.load(a_block_ptr, boundary_check=(0, 1))
        b_blck = tl.load(b_block_ptr, boundary_check=(0, 1))
        dq_b_t = dequant_4bit_body_util(
            a=b_blck,
            offsets=b_offsets,
            quant_ptr=quant_ptr,
            absmax_ptr=absmax_ptr,
            n_elems=num_paired_elements,
            QUANT_BLOCK=QUANT_BLOCK,
        )
        dq_b_t = dq_b_t.trans()
        dq_b = dq_b_t.to(a_ptr.type.element_ty)

        # We accumulate along the K dimension.
        accumulator += tl.dot(a_blck, dq_b, out_dtype=ACCUMULATOR_DTYPE)
        # Advance the ptrs to the next K block.
        b_offsets += (BLOCK_SIZE_K // 2) * stride_bk
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (0, BLOCK_SIZE_K // 2))
    c = accumulator.to(c_ptr.type.element_ty)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offset_c = bid.to(tl.int64) * stride_cz
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr + offset_c,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


# Common scenario is A batched, B is just 2d
def matmul(a, b, shapeB, code, absmax, blocksize):
    # Check constraints.
    # assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    if len(a.shape) == 1:
        B = M = 1
        K = a.shape
        stride_az, stride_am, stride_ak = a.numel(), a.numel(), a.stride(0)
    elif len(a.shape) == 2:
        B = 1
        M, K = a.shape
        stride_az, stride_am, stride_ak = a.numel(), a.stride(0), a.stride(1)
    elif len(a.shape) == 3:
        B, M, K = a.shape
        stride_az, stride_am, stride_ak = a.stride(0), a.stride(1), a.stride(2)
    elif len(a.shape) > 3:
        a = a.view(-1, a.shape[-2], a.shape[-1])
        B, M, K = a.shape
        stride_az, stride_am, stride_ak = a.stride(0), a.stride(1), a.stride(2)

    if len(shapeB) == len(a.shape) == 3:
        assert shapeB[0] == B, "Incompatible batch size"
        assert shapeB[1] == K, "Incompatible dimensions"
        B, N, K = shapeB
        stride_bz, stride_bn, stride_bk = N * K // 2, K // 2, 1
        c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)

    if len(shapeB) == 2:
        N, K = shapeB
        b = b.view(N, K // 2)
        stride_bz, stride_bn, stride_bk = N * K // 2, K // 2, 1
        if len(a.shape) >= 3:
            c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
            stride_cz, stride_cm, stride_cn = c.stride(0), c.stride(1), c.stride(2)
        else:
            c = torch.empty((M, N), device=a.device, dtype=a.dtype)
            stride_cz, stride_cm, stride_cn = c.numel(), c.stride(0), c.stride(1)

    # BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = 16, 16, 16, 4
    # Allocates output.
    # c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    # grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),B, )
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        B,
    )
    accum_dtype = a.dtype
    if a.dtype in (torch.bfloat16, torch.float16):
        accum_dtype = torch.float32

    triton_accum_dtype = tl.dtype(str(accum_dtype)[6:].replace("bfloat", "bf").replace("float", "fp"))
    # grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    number_of_paired_elements = b.numel()
    matmul_kernel[grid](
        a,
        b,
        c,  # tensors
        M,
        N,
        K,  # sizes
        stride_az,
        stride_am,
        stride_ak,  #
        stride_bz,
        stride_bn,
        stride_bk,  #
        stride_cz,
        stride_cm,
        stride_cn,  #
        code,
        absmax,
        number_of_paired_elements,
        triton_accum_dtype,
        blocksize,
        # BLOCK_SIZE_M,
        # BLOCK_SIZE_N,
        # BLOCK_SIZE_K,  #
        # GROUP_SIZE_M,
    )
    return c


@triton.jit
def matmul_kernel_quant_type(
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
    stride_az,
    stride_am,
    stride_ak,
    stride_bz,
    stride_bn,
    stride_bk,
    stride_cz,
    stride_cm,
    stride_cn,
    absmax_ptr,
    num_paired_elements,
    ACCUMULATOR_DTYPE: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    # Meta-parameters
    IS_FP4: tl.constexpr,
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
    bid = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offset_a = bid.to(tl.int64) * stride_az
    offset_b = bid.to(tl.int64) * stride_bz
    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_bk = tl.arange(0, BLOCK_SIZE_K // 2)

    b_offsets = offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr + offset_a,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr + offset_b,
        shape=(N, K // 2),
        strides=(stride_bn, stride_bk),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K // 2),
        order=(1, 0),
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUMULATOR_DTYPE)
    # dq_b = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=a_ptr.type.element_ty)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a_blck = tl.load(a_block_ptr, boundary_check=(0, 1))
        b_blck = tl.load(b_block_ptr, boundary_check=(0, 1))
        if IS_FP4 == 1:
            dq_b_t = dequant_fp4_body_util(
                a=b_blck,
                offsets=b_offsets,
                absmax_ptr=absmax_ptr,
                n_elems=num_paired_elements,
                QUANT_BLOCK=QUANT_BLOCK,
            )
        else:
            dq_b_t = dequant_nf4_body_util(
                a=b_blck,
                offsets=b_offsets,
                absmax_ptr=absmax_ptr,
                n_elems=num_paired_elements,
                QUANT_BLOCK=QUANT_BLOCK,
            )
        dq_b_t = dq_b_t.trans()
        dq_b = dq_b_t.to(a_ptr.type.element_ty)

        # We accumulate along the K dimension.
        accumulator += tl.dot(a_blck, dq_b, out_dtype=ACCUMULATOR_DTYPE)
        # Advance the ptrs to the next K block.
        b_offsets += (BLOCK_SIZE_K // 2) * stride_bk
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (0, BLOCK_SIZE_K // 2))
    c = accumulator.to(c_ptr.type.element_ty)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offset_c = bid.to(tl.int64) * stride_cz
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr + offset_c,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


def matmul_quant_typed(a, b, shapeB, quant_type, absmax, blocksize):
    flag_map = {"nf4": 0, "fp4": 1}
    flag = flag_map[quant_type]
    # Check constraints.
    # assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    if len(a.shape) == 1:
        B = M = 1
        K = a.shape
        stride_az, stride_am, stride_ak = a.numel(), a.numel(), a.stride(0)
    elif len(a.shape) == 2:
        B = 1
        M, K = a.shape
        stride_az, stride_am, stride_ak = a.numel(), a.stride(0), a.stride(1)
    elif len(a.shape) == 3:
        B, M, K = a.shape
        stride_az, stride_am, stride_ak = a.stride(0), a.stride(1), a.stride(2)
    elif len(a.shape) > 3:
        a = a.view(-1, a.shape[-2], a.shape[-1])
        B, M, K = a.shape
        stride_az, stride_am, stride_ak = a.stride(0), a.stride(1), a.stride(2)

    if len(shapeB) == len(a.shape) == 3:
        assert shapeB[0] == B, "Incompatible batch size"
        assert shapeB[1] == K, "Incompatible dimensions"
        B, N, K = shapeB
        stride_bz, stride_bn, stride_bk = N * K // 2, K // 2, 1
        c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)

    if len(shapeB) == 2:
        N, K = shapeB
        b = b.view(N, K // 2)
        stride_bz, stride_bn, stride_bk = N * K // 2, K // 2, 1
        if len(a.shape) >= 3:
            c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
            stride_cz, stride_cm, stride_cn = c.stride(0), c.stride(1), c.stride(2)
        else:
            c = torch.empty((M, N), device=a.device, dtype=a.dtype)
            stride_cz, stride_cm, stride_cn = c.numel(), c.stride(0), c.stride(1)

    # BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = 16, 16, 16, 4
    # grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), B, )
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        B,
    )
    accum_dtype = a.dtype
    if a.dtype in (torch.bfloat16, torch.float16):
        accum_dtype = torch.float32

    triton_accum_dtype = tl.dtype(str(accum_dtype)[6:].replace("bfloat", "bf").replace("float", "fp"))
    # grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    number_of_paired_elements = b.numel()
    matmul_kernel_quant_type[grid](
        a,
        b,
        c,  # tensors
        M,
        N,
        K,  # sizes
        stride_az,
        stride_am,
        stride_ak,  #
        stride_bz,
        stride_bn,
        stride_bk,  #
        stride_cz,
        stride_cm,
        stride_cn,  #
        absmax,
        number_of_paired_elements,
        triton_accum_dtype,
        blocksize,
        IS_FP4=flag,
        # BLOCK_SIZE_M,
        # BLOCK_SIZE_N,
        # BLOCK_SIZE_K,  #
        # GROUP_SIZE_M,
    )
    return c

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=2,
                      num_warps=32),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4}, num_stages=2,
                      num_warps=32),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4}, num_stages=2,
                      num_warps=32),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 4}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4}, num_stages=2,
                      num_warps=32),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 4},
            num_stages=2,
            num_warps=32,
        ),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4}, num_stages=2,
                      num_warps=32),
        # # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=2,
        # #               num_warps=32),
        # # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=3,
        # #               num_warps=32),
        # # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=2,
        # #               num_warps=32),
        # # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=2,
        # #               num_warps=32),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel_straight(
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
    stride_az,
    stride_am,
    stride_ak,
    stride_bz,
    stride_bn,
    stride_bk,
    stride_cz,
    stride_cm,
    stride_cn,
    ACCUMULATOR_DTYPE: tl.constexpr,
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
    bid = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offset_a = bid.to(tl.int64) * stride_az
    offset_b = bid.to(tl.int64) * stride_bz
    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr + offset_a,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(base=b_ptr + offset_b, shape=(K, N), strides=(stride_bk, stride_bn),
                                offsets=(0, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                                order=(1, 0))


    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUMULATOR_DTYPE)
    # dq_b = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=a_ptr.type.element_ty)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))

        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b, out_dtype=ACCUMULATOR_DTYPE)
        # Advance the ptrs to the next K block.
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))
    c = accumulator.to(c_ptr.type.element_ty)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offset_c = bid.to(tl.int64) * stride_cz
    c_block_ptr = tl.make_block_ptr(base=c_ptr + offset_c, shape=(M, N), strides=(stride_cm, stride_cn),
                                    offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


# Common scenario is A batched, B is just 2d
def triton_matmul(a, b):
    # Check constraints.
    # assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    if len(a.shape) == 1:
        B = M = 1
        K = a.shape
        stride_az, stride_am, stride_ak = a.numel(), a.numel(), a.stride(0)
    elif len(a.shape) == 2:
        B = 1
        M, K = a.shape
        stride_az, stride_am, stride_ak = a.numel(), a.stride(0), a.stride(1)
    elif len(a.shape) == 3:
        B, M, K = a.shape
        stride_az, stride_am, stride_ak = a.stride(0), a.stride(1), a.stride(2)
    elif len(a.shape) > 3:
        a = a.view(-1, a.shape[-2], a.shape[-1])
        B, M, K = a.shape
        stride_az, stride_am, stride_ak = a.stride(0), a.stride(1), a.stride(2)

    if len(b.shape) == len(a.shape) == 3:
        assert b.shape[0] == B, "Incompatible batch size"
        assert b.shape[1] == K, "Incompatible dimensions"
        B, K, N = b.shape
        stride_bz, stride_bn, stride_bk = b.stride(0), b.stride(1), b.stride(2)
        c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)

    if len(b.shape) == 2:
        K, N = b.shape
        # b = b.view(N, K // 2)
        stride_bz, stride_bn, stride_bk = b.stride(0), b.stride(1), 1
        if len(a.shape) >= 3:
            c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
            stride_cz, stride_cm, stride_cn = c.stride(0), c.stride(1), c.stride(2)
        else:
            c = torch.empty((M, N), device=a.device, dtype=a.dtype)
            stride_cz, stride_cm, stride_cn = c.numel(), c.stride(0), c.stride(1)

    # BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = 16, 16, 16, 4
    # Allocates output.
    # c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    # grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),B, )
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        B,
    )
    accum_dtype = a.dtype
    if a.dtype in (torch.bfloat16, torch.float16):
        accum_dtype = torch.float32

    triton_accum_dtype = tl.dtype(str(accum_dtype)[6:].replace("bfloat", "bf").replace("float", "fp"))
    # grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel_straight[grid](
        a,
        b,
        c,  # tensors
        M,
        N,
        K,  # sizes
        stride_az,
        stride_am,
        stride_ak,  #
        stride_bz,
        stride_bn,
        stride_bk,  #
        stride_cz,
        stride_cm,
        stride_cn,  #
        triton_accum_dtype,
        # BLOCK_SIZE_M,
        # BLOCK_SIZE_N,
        # BLOCK_SIZE_K,  #
        # GROUP_SIZE_M,
    )
    return c


######################### Fallback dequantization functions #########################
## for debug ##


# @triton.autotune(
#     configs=[
#         # triton.Config({'SPLIT_NUM_BLOCKS': 1, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
#         # triton.Config({'SPLIT_NUM_BLOCKS': 1, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
#         # triton.Config({'SPLIT_NUM_BLOCKS': 1, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
#         # #
#         # triton.Config({"SPLIT_NUM_BLOCKS": 1, "grf_mode": "auto"}, num_stages=4, num_warps=32),
#         #
#         triton.Config({"SPLIT_NUM_BLOCKS": 2}),
#         # triton.Config({"SPLIT_NUM_BLOCKS": 2, "grf_mode": "large"}, num_stages=2, num_warps=32),
#         # # triton.Config({'SPLIT_NUM_BLOCKS': 2, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
#         # triton.Config({"SPLIT_NUM_BLOCKS": 2, "grf_mode": "auto"}, num_stages=2, num_warps=32),
#         # triton.Config({"SPLIT_NUM_BLOCKS": 2, "grf_mode": "auto"}, num_stages=4, num_warps=32),
#         # triton.Config({"SPLIT_NUM_BLOCKS": 4, "grf_mode": "large"}, num_stages=2, num_warps=32),
#         # triton.Config({"SPLIT_NUM_BLOCKS": 4, "grf_mode": "large"}, num_stages=4, num_warps=32),
#         # triton.Config({'SPLIT_NUM_BLOCKS': 8, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
#     ],
#     key=["n_elements", "BLOCK_SIZE"],
# )
@triton.jit
def quantize_4bit_blockwise_kernel(
    A_ptr,
    code_ptr,
    absmax_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    CODE_SIZE: tl.constexpr,
    SPLIT_NUM_BLOCKS: tl.constexpr,
):
    PAIRED_SPLIT_NUM_BLOCKS: tl.constexpr = SPLIT_NUM_BLOCKS * 2
    block_start_idx = tl.program_id(0) * PAIRED_SPLIT_NUM_BLOCKS
    thread_idx = tl.arange(0, PAIRED_SPLIT_NUM_BLOCKS * BLOCK_SIZE)

    offsets = block_start_idx * BLOCK_SIZE + thread_idx
    mask = offsets < n_elements

    A = tl.load(A_ptr + offsets, mask=mask, other=0.0)

    # To be able process several blocks -> (PAIRED_SPLIT_NUM_BLOCKS, BLOCK_SIZE)
    A_reshaped = tl.reshape(A, (PAIRED_SPLIT_NUM_BLOCKS, BLOCK_SIZE))

    # Calculating absamax for each block
    absmax = tl.max(tl.abs(A_reshaped), axis=1)
    tl.store(absmax_ptr + block_start_idx + tl.arange(0, PAIRED_SPLIT_NUM_BLOCKS), absmax)

    A_normalized = A_reshaped / absmax[:, None]
    A_normalized = tl.clamp(A_normalized, -1.0, 1.0)

    lower_pivot = tl.zeros((PAIRED_SPLIT_NUM_BLOCKS, BLOCK_SIZE), dtype=tl.int32)
    upper_pivot = tl.full((PAIRED_SPLIT_NUM_BLOCKS, BLOCK_SIZE), CODE_SIZE - 1, dtype=tl.int32)

    for _ in range(4):  # ceil(log2(code_size)) = 4, actually, in general case should be input parameter
        pivot = (lower_pivot + upper_pivot) // 2
        val = tl.load(code_ptr + pivot)
        is_higher = A_normalized > val  # code[pivot]
        lower_pivot = tl.where(is_higher, pivot, lower_pivot)
        upper_pivot = tl.where(is_higher, upper_pivot, pivot)

    # Choose closest level
    lower_val = tl.load(code_ptr + lower_pivot)
    upper_val = tl.load(code_ptr + upper_pivot)
    lower_dist = tl.abs(A_normalized - lower_val)
    upper_dist = tl.abs(A_normalized - upper_val)
    quantized = tl.where(lower_dist <= upper_dist, lower_pivot, upper_pivot).to(tl.uint8)

    quantized = quantized.reshape((PAIRED_SPLIT_NUM_BLOCKS, BLOCK_SIZE // 2, 2))
    quantized = quantized.to(tl.uint8, bitcast=True)
    left, right = quantized.split()
    packed = left << 4 | (right & 0xF)

    # Reduce don't guarantee the order of the elements passed to unite_2_int4
    # packed = tl.reduce(quantized, axis=2, combine_fn=unite_2_int4)
    # packed = packed.to(tl.uint8, bitcast=True)

    packed_flat = tl.reshape(packed, (BLOCK_SIZE * SPLIT_NUM_BLOCKS,))
    out_offsets = block_start_idx * BLOCK_SIZE // 2 + tl.arange(0, SPLIT_NUM_BLOCKS * BLOCK_SIZE)
    out_mask = out_offsets < n_elements // 2
    tl.store(out_ptr + out_offsets, packed_flat, mask=out_mask)

@triton.jit
def dequant_8bit_kernel_util(
    a,
    offsets,
    quant_ptr,
    absmax_ptr,
    num_paired_elements,
    QUANT_BLOCK: tl.constexpr,
):
    mask = offsets < num_paired_elements

    abs_offsets = offsets // QUANT_BLOCK
    absmax = tl.load(absmax_ptr + abs_offsets, mask=mask, other=1.0, eviction_policy="evict_last")

    # apply conversion
    scaled_int8 = tl.load(quant_ptr + a, mask)
    # apply scales
    out_dq = scaled_int8 * absmax
    return out_dq

@triton.jit
def quantize_8bit_blockwise_kernel_util(
    A_ptr,
    code_ptr,
    absmax_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    CODE_SIZE: tl.constexpr,
    SPLIT_NUM_BLOCKS: tl.constexpr,
):
    block_start_idx = tl.program_id(0) * SPLIT_NUM_BLOCKS
    thread_idx = tl.arange(0, SPLIT_NUM_BLOCKS * BLOCK_SIZE)

    offsets = block_start_idx * BLOCK_SIZE + thread_idx
    mask = offsets < n_elements

    A = tl.load(A_ptr + offsets, mask=mask, other=0.0)

    # To be able process several blocks -> (BLOCK_SIZE, SPLIT_NUM_BLOCKS)
    A_reshaped = tl.reshape(A, (SPLIT_NUM_BLOCKS, BLOCK_SIZE))

    # Calculating absamax for each block
    absmax = tl.max(tl.abs(A_reshaped), axis=1)
    tl.store(absmax_ptr + block_start_idx + tl.arange(0, SPLIT_NUM_BLOCKS), absmax)

    A_normalized = A_reshaped / absmax[:, None]
    A_normalized = tl.clamp(A_normalized, -1.0, 1.0)

    lower_pivot = tl.zeros((SPLIT_NUM_BLOCKS, BLOCK_SIZE), dtype=tl.int32)
    upper_pivot = tl.full((SPLIT_NUM_BLOCKS, BLOCK_SIZE), CODE_SIZE - 1, dtype=tl.int32)

    for _ in range(8):  # ceil(log2(code_size)) = 8, actually, in general case should be input parameter
        pivot = (lower_pivot + upper_pivot) // 2
        val = tl.load(code_ptr + pivot)
        is_higher = A_normalized > val  # code[pivot]
        lower_pivot = tl.where(is_higher, pivot, lower_pivot)
        upper_pivot = tl.where(is_higher, upper_pivot, pivot)

    # Choose closest level
    lower_val = tl.load(code_ptr + lower_pivot)
    upper_val = tl.load(code_ptr + upper_pivot)
    lower_dist = tl.abs(A_normalized - lower_val)
    upper_dist = tl.abs(A_normalized - upper_val)
    quantized = tl.where(lower_dist <= upper_dist, lower_pivot, upper_pivot).to(tl.uint8)

    # too slow approach
    # diff = tl.abs(A_normalized[:, :, None] - code[None, None, :])
    # quantized = tl.argmin(diff, axis=2).to(tl.uint8)

    quantized_flat = tl.reshape(quantized, (BLOCK_SIZE * SPLIT_NUM_BLOCKS,))
    tl.store(out_ptr + offsets, quantized_flat, mask=mask)

@triton.jit
def quantize_2d(x, quadrants_ptr, code_ptr, SIGNED: tl.constexpr):
    # x: tl.scalar or tl.tensor
    # quadrants: tl.tensor, shape [3]
    # smem_code: tl.tensor, shape [256]
    # Возвращает: индекс (uint32)

    pivot = tl.zeros_like(x).to(tl.int32) + 127
    upper_pivot = tl.zeros_like(x).to(tl.int32) + 255
    lower_pivot = tl.zeros_like(x).to(tl.int32)
    lower = tl.zeros_like(x) + (-1.0 if SIGNED else 0.0)
    upper = tl.zeros_like(x) + 1.0
    val = tl.zeros_like(x) + tl.load(quadrants_ptr + 1)
    local_pivot = tl.zeros_like(x).to(tl.int32) + 1
    offset = tl.zeros_like(x).to(tl.int32) + 1

    i = 64
    while i > 0:
        is_higher = x > val
        lower_pivot = tl.where(is_higher, pivot, lower_pivot)
        upper_pivot = tl.where(is_higher, upper_pivot, pivot)
        lower = tl.where(is_higher, val, lower)
        upper = tl.where(is_higher, upper, val)
        pivot = tl.where(is_higher, pivot + i, pivot - i)
        local_pivot = tl.where(is_higher, local_pivot + offset, local_pivot - offset)
        quadrants_val = tl.load(quadrants_ptr + lower_pivot)
        code_val = tl.load(code_ptr + upper_pivot)
        val = tl.where(i >= 64, quadrants_val, code_val)
        offset = offset - 1
        i = i // 2

    is_higher = x > val
    midpoint = tl.where(is_higher, (upper + val) * 0.5, (lower + val) * 0.5)
    result = tl.where(is_higher, tl.where(x > midpoint, upper_pivot, pivot), tl.where(x < midpoint, lower_pivot, pivot))
    return result.to(tl.uint32)

# @triton.jit
# def optimizer_static8bit2state_blockwise_kernel(
#     p_ptr, g_ptr, state1_ptr, state2_ptr,
#     beta1, beta2, beta3, alpha, eps, step, lr,
#     quantiles1_ptr, quantiles2_ptr,
#     absmax1_ptr, absmax2_ptr,
#     weight_decay, gnorm_scale, skip_zeros,
#     n,
#     BLOCK_SIZE: tl.constexpr,
#     N_PER_TH: tl.constexpr,
# ):
#     block_idx = tl.program_id(0)
#     thread_idx = tl.arange(0, BLOCK_SIZE)
#     base_idx = block_idx * BLOCK_SIZE

#     # Индексы для этого блока
#     idx = base_idx + thread_idx

#     # Загрузка quantiles в локальные массивы (имитация shared memory)
#     quantiles1 = tl.load(quantiles1_ptr + thread_idx, mask=thread_idx < 256, other=0.0)
#     quantiles2 = tl.load(quantiles2_ptr + thread_idx, mask=thread_idx < 256, other=0.0)

#     # Для каждого потока: обработка N_PER_TH элементов
#     for j in range(N_PER_TH):
#         i = idx + j * BLOCK_SIZE
#         mask = i < n

#         # Загрузка градиентов, состояний, параметров
#         g_val = tl.load(g_ptr + i, mask=mask, other=0.0)
#         p_val = tl.load(p_ptr + i, mask=mask, other=0.0)
#         c1 = tl.load(state1_ptr + i, mask=mask, other=0)
#         c2 = tl.load(state2_ptr + i, mask=mask, other=0)

#         # Деквантование состояний
#         # NOT sure abount block size
#         c1_dq = dequant_8bit_kernel_util(c1, i, quantiles1_ptr, absmax1_ptr + (i // BLOCK_SIZE), n, BLOCK_SIZE)
#         c2_dq = dequant_8bit_kernel_util(c2, i, quantiles2_ptr, absmax1_ptr + (i // BLOCK_SIZE), n, BLOCK_SIZE)
#         s1 = c1_dq * tl.load(absmax1_ptr + (i // BLOCK_SIZE))
#         s2 = c2_dq * tl.load(absmax2_ptr + (i // BLOCK_SIZE))

#         # Обновление состояний
#         g_val = g_val * gnorm_scale
#         s2 = (s2 * beta2) + ((1.0 - beta2) * g_val * g_val)
#         s1 = (s1 * beta1) + ((1.0 - beta1) * g_val)

#         # Обновление параметров
#         correction1 = 1.0 - tl.math.pow(beta1, step)
#         correction2 = tl.math.sqrt(1.0 - tl.math.pow(beta2, step))
#         step_size = -lr * correction2 / correction1
#         p_val = p_val + (step_size * (s1 / (tl.math.sqrt(s2) + (correction2 * eps))))
#         if weight_decay > 0.0:
#             p_val = p_val * (1.0 - (lr * weight_decay))

#         # Квантование состояний обратно
#         # (Здесь предполагается, что quantize_2D реализован отдельно)
#         # c1_new = quantize_2D(quadrants1, quantiles1, s1 / new_local_abs_max1)
#         c1_new = quantize_2d(s1 / new_local_abs_max1, quadrants1, quantiles1)
#         # c2_new = quantize_2D(quadrants2, quantiles2, s2 / new_local_abs_max2)
#         # Для простоты: просто округляем к ближайшему индексу
#         c1_new = tl.math.min(255, tl.math.max(0, tl.math.round(s1 / tl.load(absmax1_ptr + (i // BLOCK_SIZE)) * 255)))
#         c2_new = tl.math.min(255, tl.math.max(0, tl.math.round(s2 / tl.load(absmax2_ptr + (i // BLOCK_SIZE)) * 255)))

#         # Сохраняем результаты
#         tl.store(p_ptr + i, p_val, mask=mask)
#         tl.store(state1_ptr + i, c1_new.to(tl.uint8), mask=mask)
#         tl.store(state2_ptr + i, c2_new.to(tl.uint8), mask=mask)

@triton.jit
def optimizer_static8bit2state_blockwise_kernel(
    p_ptr, g_ptr, state1_ptr, state2_ptr,
    beta1: tl.constexpr, 
    beta2: tl.constexpr, 
    beta3: tl.constexpr, 
    alpha: tl.constexpr, 
    eps: tl.constexpr, 
    step: tl.constexpr, 
    lr: tl.constexpr,
    quantiles1_ptr, quantiles2_ptr,
    absmax1_ptr, absmax2_ptr,
    weight_decay, gnorm_scale, skip_zeros,
    n,
    BLOCK_SIZE: tl.constexpr,
    N_PER_TH: tl.constexpr,
    SIGNED: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n
    QUAD: tl.constexpr = 3

    # Загрузка quantiles и quadrants в локальные переменные
    smem_code1 = tl.load(quantiles1_ptr + tl.arange(0, 256))
    smem_code2 = tl.load(quantiles2_ptr + tl.arange(0, 256))
    quant1_wa = tl.arange(0, 4)
    mask_wa = quant1_wa < QUAD
    quadrants1 = tl.load(quantiles1_ptr + quant1_wa, mask=mask_wa, other=0.0)
    quadrants2 = tl.load(quantiles2_ptr + quant1_wa, mask=mask_wa, other=0.0)

    # Загрузка параметров, градиентов и состояний
    p = tl.load(p_ptr + idx, mask=mask)
    g = tl.load(g_ptr + idx, mask=mask)
    c1 = tl.load(state1_ptr + idx, mask=mask).to(tl.uint8)
    c2 = tl.load(state2_ptr + idx, mask=mask).to(tl.uint8)

    absmax1 = tl.load(absmax1_ptr + (idx // BLOCK_SIZE))
    absmax2 = tl.load(absmax2_ptr + (idx // BLOCK_SIZE))

    # Деквантование состояний
    c1_dq = dequant_8bit_kernel_util(c1, idx, quantiles1_ptr, absmax1_ptr + (idx // BLOCK_SIZE), n, BLOCK_SIZE)
    c2_dq = dequant_8bit_kernel_util(c2, idx, quantiles2_ptr, absmax1_ptr + (idx // BLOCK_SIZE), n, BLOCK_SIZE)
    s1 = c1_dq * absmax1
    s2 = c2_dq * absmax2

    # Обновление состояний
    g_scaled = g * gnorm_scale
    s2_new = s2 * beta2 + (1.0 - beta2) * g_scaled * g_scaled
    s1_new = s1 * beta1 + (1.0 - beta1) * g_scaled

    # Обновление параметров
    correction1 = 1.0 - (beta1 ** step)
    correction2 = tl.sqrt(1.0 - beta2 ** step)
    step_size = -lr * correction2 / correction1
    p_new = p + (step_size * (s1_new / (tl.sqrt(s2_new) + (correction2 * eps))))
    if weight_decay > 0.0:
        p_new = p_new * (1.0 - (lr * weight_decay))

    # Квантование состояний обратно с помощью quantize_2d
    s1_norm = s1_new / tl.max(tl.abs(s1_new), axis=0)
    s2_norm = s2_new / tl.max(tl.abs(s2_new), axis=0)
    c1_new = quantize_2d(s1_norm, quantiles1_ptr, quantiles1_ptr , SIGNED)
    c2_new = quantize_2d(s2_norm, quantiles2_ptr, quantiles2_ptr, 0)

    # Сохраняем результаты
    tl.store(p_ptr + idx, p_new, mask=mask)
    tl.store(state1_ptr + idx, c1_new, mask=mask)
    tl.store(state2_ptr + idx, c2_new, mask=mask)

def optim_kernel_call(
        p: torch.Tensor,
        g: torch.Tensor,
        state1: torch.Tensor,
        state2: torch.Tensor,
        beta1: float,
        beta2: float,
        beta3: float,
        alpha: float,
        eps: float,
        step: int,
        lr: float,
        qmap1: torch.Tensor,
        qmap2: torch.Tensor,
        absmax1: torch.Tensor,
        absmax2: torch.Tensor,
        weight_decay: float = 0.0,
        gnorm_scale: float = 1.0,
        skip_zeros=False,
        n: int = 0,
):
    BLOCK_SIZE = 256
    N_PER_TH = 1
    grid = lambda META: (triton.cdiv(n, BLOCK_SIZE * N_PER_TH),)
    print("Using Triton kernel")
    optimizer_static8bit2state_blockwise_kernel[grid](
        p, g, state1, state2,
        beta1, beta2, beta3, alpha, eps, step, lr,
        qmap1, qmap2,
        absmax1, absmax2,
        weight_decay, gnorm_scale, skip_zeros,
        n,
        BLOCK_SIZE, N_PER_TH
    )