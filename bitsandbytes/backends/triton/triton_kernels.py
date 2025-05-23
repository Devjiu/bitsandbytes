from collections.abc import Sequence

import torch

from bitsandbytes.functional import get_4bit_type
import triton
import triton.language as tl

# Should be the same for quant/dequant
_FP4_QUANT_TABLE = get_4bit_type("fp4", device="xpu")
_NF4_QUANT_TABLE = get_4bit_type("nf4", device="xpu")


@triton.autotune(
    configs=[
        # triton.Config({'SPLIT_SIZE': 64}),
        # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
        # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
        # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
        # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'auto'}, num_stages=4, num_warps=32),
        # triton.Config({'SPLIT_SIZE': 128}),
        # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
        # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
        # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
        # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'auto'}, num_stages=4, num_warps=32),
        triton.Config({"SPLIT_SIZE": 256}),
        # triton.Config({'SPLIT_SIZE': 256, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
        # triton.Config({'SPLIT_SIZE': 256, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
        triton.Config({"SPLIT_SIZE": 512}),
        # triton.Config({'SPLIT_SIZE': 1024}),
    ],
    key=["num_paired_elements", "QUANT_BLOCK"],
)
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
    # code = tl.load(quant_ptr + tl.arange(0, 256))
    # # Gather the values from the codebook using the indices in 'a'
    # scaled_int8 = tl.gather(code, a, axis=0)

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


# @triton.jit
# def dequant_8bit_kernel(
#     a_ptr,
#     c_ptr,
#     quant_ptr,
#     absmax_ptr,
#     num_paired_elements,
#     QUANT_BLOCK: tl.constexpr,
#     SPLIT_SIZE: tl.constexpr,
# ):
#     pid = tl.program_id(axis=0)
#     block_start = pid * SPLIT_SIZE
#     offsets = block_start + tl.arange(0, SPLIT_SIZE)
#     mask = offsets < num_paired_elements

#     a = tl.load(a_ptr + offsets, mask)

#     %9 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>> loc(#loc9)
#     %10 = tt.addptr %9, %5 : tensor<512x!tt.ptr<f32>>, tensor<512xi32> loc(#loc9)
#     %11 = tt.load %10 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>> loc(#loc10)
#     %12 = arith.extui %8 : tensor<512xi8> to tensor<512xi32> loc(#loc11)
#     %13 = arith.addi %12, %cst_0 : tensor<512xi32> loc(#loc12)
#     %14 = arith.cmpi slt, %12, %cst : tensor<512xi32> loc(#loc13)
#     %15 = arith.select %14, %13, %12 : tensor<512xi1>, tensor<512xi32> loc(#loc14)
#     %16 = arith.cmpi sge, %15, %cst : tensor<512xi32> loc(#loc15)
#     %17 = arith.cmpi slt, %15, %cst_0 : tensor<512xi32> loc(#loc16)
#     %18 = arith.andi %16, %17 : tensor<512xi1> loc(#loc17)
#     tt.assert %18, "index out of bounds: 0 <= tmp5 < 256" : tensor<512xi1> loc(#loc18)
#     %19 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>> loc(#loc19)
#     %20 = tt.addptr %19, %15 : tensor<512x!tt.ptr<f32>>, tensor<512xi32> loc(#loc19)
#     %21 = tt.load %20 evictionPolicy = evict_last : tensor<512x!tt.ptr<f32>> loc(#loc20)
#     %22 = arith.mulf %21, %11 : tensor<512xf32> loc(#loc21)
#     %23 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>> loc(#loc22)
#     %24 = tt.addptr %23, %4 : tensor<512x!tt.ptr<f32>>, tensor<512xi32> loc(#loc22)
#     tt.store %24, %22 : tensor<512x!tt.ptr<f32>> loc(#loc23)

# @triton_heuristics.pointwise(
#     size_hints={'x': 1048576},
#     filename=__file__,
#     triton_meta={'signature': {'in_ptr0': '*u8', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='xpu', index=0, multi_processor_count=56, cc={'architecture': 13136561920, 'driver_version': '1.6.32567+18', 'gpu_eu_count': 448, 'gpu_subslice_count': 56, 'has_atomic64': True, 'has_bfloat16_conversions': True, 'has_fp16': True, 'has_fp64': True, 'has_subgroup_2d_block_io': True, 'has_subgroup_matrix_multiply_accumulate': True, 'has_subgroup_matrix_multiply_accumulate_tensor_float32': False, 'max_compute_units': 448, 'max_num_sub_groups': 64, 'max_work_group_size': 1024, 'name': 'Intel(R) Data Center GPU Max 1100', 'platform_name': 'Intel(R) oneAPI Unified Runtime over Level-Zero', 'sub_group_sizes': [16, 32], 'total_memory': 51539607552, 'type': 'gpu', 'vendor': 'Intel(R) Corporation', 'version': '12.60.7'}, major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
#     inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '0917572EF3401D0AE4F8A603B8359B7C8E91B071FC4584D13742217117B09AAD', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
#     min_elem_per_thread=0
# )
# @triton.jit
# def triton_poi_fused_mul_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
#     xnumel = 1048576
#     xoffset = tl.program_id(0) * XBLOCK
#     xindex = xoffset + tl.arange(0, XBLOCK)[:]
#     xmask = tl.full([XBLOCK], True, tl.int1)
#     x2 = xindex
#     x1 = xindex // 256
#     tmp0 = tl.load(in_ptr0 + (x2), None)
#     tmp8 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
#     tmp1 = tmp0.to(tl.int32)
#     tmp2 = tl.full([XBLOCK], 256, tl.int32)
#     tmp3 = tmp1 + tmp2
#     tmp4 = tmp1 < 0
#     tmp5 = tl.where(tmp4, tmp3, tmp1)
#     tl.device_assert((0 <= tmp5) & (tmp5 < 256), "index out of bounds: 0 <= tmp5 < 256")
#     tmp7 = tl.load(in_ptr1 + (tmp5), None, eviction_policy='evict_last')
#     tmp9 = tmp7 * tmp8
#     tl.store(out_ptr0 + (x2), tmp9, None)


@triton.autotune(
    configs=[
        # triton.Config({'SPLIT_SIZE': 64}),
        # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
        # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
        # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
        # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'auto'}, num_stages=4, num_warps=32),
        # triton.Config({'SPLIT_SIZE': 128}),
        # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
        # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
        # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
        # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'auto'}, num_stages=4, num_warps=32),
        # triton.Config({'SPLIT_SIZE': 256}),
        # triton.Config({'SPLIT_SIZE': 256, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
        # triton.Config({'SPLIT_SIZE': 256, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
        triton.Config({"SPLIT_SIZE": 512}),
        # triton.Config({'SPLIT_SIZE': 1024}),
    ],
    key=["xnumel", "QUANT_BLOCK"],
)
@triton.jit
def dequantize_kernel(a_ptr, code_ptr, absmax_ptr, c_ptr, xnumel, QUANT_BLOCK: tl.constexpr, SPLIT_SIZE: tl.constexpr):
    # a_ptr,
    # c_ptr,
    # quant_ptr,
    # absmax_ptr,
    # num_paired_elements,
    # QUANT_BLOCK: tl.constexpr,
    # SPLIT_SIZE: tl.constexpr,
    """
    Dequantizes a blockwise quantized tensor.

    Args:
        a_ptr: Pointer to the quantized data (uint8).
        code_ptr: Pointer to the quantization codebook (float32).
        absmax_ptr: Pointer to the absmax values (float32).
        c_ptr: Pointer to the output tensor (float32).
        xnumel: Number of elements to process.
        XBLOCK: Block size for Triton.
    """
    xoffset = tl.program_id(0) * SPLIT_SIZE
    xindex = xoffset + tl.arange(0, SPLIT_SIZE)
    xmask = xindex < xnumel  # Ensure we don't go out of bounds

    # Load the quantized data
    a_quantized_indices = tl.load(a_ptr + xindex, mask=xmask, other=0).to(tl.int32)

    # Calculate block index and absmax offset
    block_index = xindex // QUANT_BLOCK
    # is_valid_block = block_index < (xnumel // QUANT_BLOCK)

    # Load the absmax value for the block
    # abs_offsets = xindex // QUANT_BLOCK
    absmax = tl.load(absmax_ptr + block_index, mask=xmask, other=0, eviction_policy="evict_last")

    # Load the dequantized values from the codebook
    # Handle potential out-of-bounds indices
    oob_mask = (a_quantized_indices < 0) | (a_quantized_indices >= 256)
    a_quantized_indices = tl.where(oob_mask, tl.zeros(a_quantized_indices.shape, tl.int32), a_quantized_indices)

    dequantized_values = tl.load(code_ptr + a_quantized_indices, mask=xmask, other=0, eviction_policy="evict_last")

    # Scale the dequantized values by absmax
    output_values = dequantized_values * absmax

    # Store the results
    tl.store(c_ptr + xindex, output_values, mask=xmask)


def dequant_int8_blockwise(
    A_nf4: torch.Tensor,
    quant_state_code: torch.Tensor,
    absmax: torch.Tensor,
    out: torch.Tensor,
    quant_blocksize: int = 64,
):
    number_of_paired_elements = A_nf4.numel()

    # SPLIT_SIZE = 256
    grid = lambda META: (triton.cdiv(number_of_paired_elements, META["SPLIT_SIZE"]),)
    # grid = (triton.cdiv(number_of_paired_elements, SPLIT_SIZE),)
    # dequantize_kernel[grid](
    #     A_nf4,
    #     quant_state_code,
    #     absmax,
    #     out,
    #     number_of_paired_elements,
    #     quant_blocksize,
    # )
    dequant_8bit_kernel[grid](
        A_nf4,
        out,
        quant_state_code,
        absmax,
        number_of_paired_elements,
        quant_blocksize,  # SPLIT_SIZE
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({"SPLIT_NUM_BLOCKS": 1, "grf_mode": "auto"}, num_stages=4, num_warps=32),
    ],
    key=["BLOCK_SIZE"],
)
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

    # grid = (triton.cdiv(blocks, split_num_blocks),)
    grid = lambda META: (triton.cdiv(blocks, META["SPLIT_NUM_BLOCKS"]),)
    quantize_blockwise_kernel[grid](
        A_ptr=A,
        code_ptr=code,
        absmax_ptr=absmax,
        out_ptr=quantized_out,
        n_elements=n,
        BLOCK_SIZE=blocksize,
        CODE_SIZE=code.numel(),
        # SPLIT_NUM_BLOCKS=split_num_blocks,
    )

    return quantized_out, absmax


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

    cond1 = A_normalized > 0.03979014977812767
    cond2 = A_normalized > 0.3893125355243683
    cond3 = A_normalized > 0.6427869200706482
    cond4 = A_normalized > 0.8614784181118011
    cond5 = A_normalized > 0.5016634166240692
    cond6 = A_normalized > 0.2035212516784668
    cond7 = A_normalized > 0.2920137718319893
    cond8 = A_normalized > 0.1202552504837513
    cond9 = A_normalized > -0.33967943489551544
    cond10 = A_normalized > -0.13791173323988914
    cond11 = A_normalized > -0.045525018125772476
    cond12 = A_normalized > -0.23460740596055984
    cond13 = A_normalized > -0.6106329262256622
    cond14 = A_normalized > -0.4599952697753906
    cond15 = A_normalized > -0.8480964004993439

    branch1 = tl.where(
        cond2,
        tl.where(cond3, tl.where(cond4, 0b1111, 0b1110), tl.where(cond5, 0b1101, 0b1100)),
        tl.where(cond6, tl.where(cond7, 0b1011, 0b1010), tl.where(cond8, 0b1001, 0b1000)),
    )
    branch2 = tl.where(
        cond9,
        tl.where(cond10, tl.where(cond11, 0b0111, 0b0110), tl.where(cond12, 0b0101, 0b0100)),
        tl.where(cond13, tl.where(cond14, 0b0011, 0b0010), tl.where(cond15, 0b0001, 0b0000)),
    )
    result = tl.where(cond1, branch1, branch2)
    quantized = result.to(tl.uint8)

    quantized = quantized.reshape((PAIRED_SPLIT_NUM_BLOCKS, BLOCK_SIZE // 2, 2))

    left, right = quantized.split()
    packed = left << 4 | (right & 0xF)
    # packed = packed.to(tl.uint8, bitcast=True)

    packed_flat = tl.reshape(packed, (BLOCK_SIZE * SPLIT_NUM_BLOCKS,))
    out_offsets = block_start_idx * BLOCK_SIZE // 2 + tl.arange(0, SPLIT_NUM_BLOCKS * BLOCK_SIZE)
    out_mask = out_offsets < n_elements // 2
    tl.store(out_ptr + out_offsets, packed_flat, mask=out_mask)


def quantize_4bit_blockwise_triton(A, blocksize, quant_type, blocks, absmax, num_elements, quantized_out):
    # grid = lambda META: (triton.cdiv(blocks, META["SPLIT_NUM_BLOCKS"]),)
    split_num_blocks = 1
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
    PAIRED_QUANT_BLOCK:tl.constexpr = QUANT_BLOCK // 2
    mask = offsets < n_elems
    higher = a & 0xF
    # print("higher: ", higher)
    # lower 4bits
    lower = a >> 4
    # print("lower: ", lower)

    # abs_blocks_lim = (
    #     num_paired_elements // PAIRED_QUANT_BLOCK
    # ) * PAIRED_QUANT_BLOCK + num_paired_elements % PAIRED_QUANT_BLOCK
    # abs_offsets = offsets // PAIRED_QUANT_BLOCK
    # mask_blocked = offsets < abs_blocks_lim
    # absmax = tl.load(absmax_ptr + abs_offsets, mask_blocked, eviction_policy="evict_last")
    abs_offsets = offsets // PAIRED_QUANT_BLOCK
    absmax = tl.load(absmax_ptr + abs_offsets, mask=mask, other=1.0, eviction_policy="evict_last")

    # out_block_start = pid * SPLIT_SIZE * 2
    # offs_low = out_block_start + 2 * tl.arange(0, SPLIT_SIZE)
    # offs_high = offs_low + 1

    # shuffle idx типа должен тут работать, мб inline asm
    # %cst0 = arith.constant 0 : i32
    # %7, %8 = gpu.shuffle idx %0, %cst0, %width : f32

    # apply conversion
    lower_4 = tl.load(quant_ptr + lower, eviction_policy="evict_last")
    higher_4 = tl.load(quant_ptr + higher, eviction_policy="evict_last")
    # print("lower uint: ", lower.view(8, 32))
    # print("lower     : ", lower_4.view(8, 32))

    # out_ref = tl.interleave(lower_4, higher_4)
    # print("out_ref: ", out_ref)
    mul_high = higher_4 * absmax
    mul_low = lower_4 * absmax
    out_dq = tl.interleave(mul_low, mul_high)
    return out_dq


# @triton.autotune(
#     configs=[
#         # triton.Config({'SPLIT_SIZE': 64}),
#         # # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
#         # # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
#         # # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
#         # # triton.Config({'SPLIT_SIZE': 64, 'grf_mode': 'auto'}, num_stages=4, num_warps=32),
#         # triton.Config({'SPLIT_SIZE': 128}),
#         # triton.Config({'SPLIT_SIZE': 128}, num_warps = 8, num_stages = 4),
#         # triton.Config({'SPLIT_SIZE': 128}, num_warps = 4, num_stages = 4),
#         # # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
#         # # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
#         # # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
#         # # triton.Config({'SPLIT_SIZE': 128, 'grf_mode': 'auto'}, num_stages=4, num_warps=32),
#         # triton.Config({'SPLIT_SIZE': 256}),
#         # triton.Config({'SPLIT_SIZE': 256}, num_warps = 8, num_stages = 4),
#         triton.Config({'SPLIT_SIZE': 256}, num_warps = 4, num_stages = 4),
#         # triton.Config({'SPLIT_SIZE': 512}),
#         # triton.Config({'SPLIT_SIZE': 512}, num_warps = 8, num_stages = 4),
#         # triton.Config({'SPLIT_SIZE': 512}, num_warps = 4, num_stages = 4),
#         # # triton.Config({'SPLIT_SIZE': 512, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
#         # # triton.Config({'SPLIT_SIZE': 512, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
#         # # triton.Config({'SPLIT_SIZE': 512, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
#         # # triton.Config({'SPLIT_SIZE': 512, 'grf_mode': 'auto'}, num_stages=4, num_warps=32),
#         # triton.Config({'SPLIT_SIZE': 1024}),
#         # # triton.Config({'SPLIT_SIZE': 2048}),
#         # # triton.Config({'SPLIT_SIZE': 4096}),
#         # # triton.Config({'SPLIT_SIZE': 8192}),
#         # # triton.Config({'SPLIT_SIZE': 16384}),
#     ],
#     key=['num_paired_elements', 'QUANT_BLOCK'],
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

    out_dq = dequant_4bit_body_util(a=a, offsets=offsets, quant_ptr=quant_ptr, absmax_ptr=absmax_ptr, n_elems=num_paired_elements, QUANT_BLOCK=QUANT_BLOCK)

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

    SPLIT_SIZE = 512
    # grid = lambda META: (triton.cdiv(number_of_paired_elements, META['SPLIT_SIZE']), )
    grid = (triton.cdiv(number_of_paired_elements, SPLIT_SIZE),)
    if quant_type == "fp4":
        dequant_4bit_kernel[grid](A, out, _FP4_QUANT_TABLE, absmax, number_of_paired_elements, blocksize, SPLIT_SIZE)
    else:
        dequant_4bit_kernel[grid](A, out, _NF4_QUANT_TABLE, absmax, number_of_paired_elements, blocksize, SPLIT_SIZE)


# from torch.library import register_fake, register_kernel
# # my_lib = torch.library.Library("bitsandbytes", "DEF")

# torch.library.define(
#     "bitsandbytes::_dequantize_4bit_impl_passing_code",
#     "(Tensor A, Tensor absmax, int blocksize, Tensor code, ScalarType dtype, Tensor out) -> None",
# )


# # @my_lib.impl("_dequantize_4bit_impl_passing_code", "meta")
# @register_fake("bitsandbytes::_dequantize_4bit_impl_passing_code")
# def _dequantize_4bit_impl_passing_code_fake(
#     A: torch.Tensor,
#     absmax: torch.Tensor,
#     blocksize: int,
#     code: torch.Tensor,
#     dtype: torch.dtype,
#     out: torch.Tensor,
# ) -> None:
#     # Просто заполняем выходной тензор "out" пустыми значениями нужного типа и формы
#     out_shape = out.shape if out is not None else A.shape
#     out_fake = torch.empty(out_shape, dtype=dtype, device='meta')
#     if out is not None:
#         out.copy_(out_fake)
#     else:
#         return out_fake


# @register_kernel("bitsandbytes::_dequantize_4bit_impl_passing_code", "xpu")
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
    # print("dequant code: ", code)
    # grid = lambda META: (triton.cdiv(number_of_paired_elements, META['SPLIT_SIZE']), )
    grid = (triton.cdiv(number_of_paired_elements, SPLIT_SIZE),)
    dequant_4bit_kernel[grid](A, out, code, absmax, number_of_paired_elements, blocksize, SPLIT_SIZE)


# ==================================================


# experimental


@torch.compile
def dequant_4bit_blockwise(
    A: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    code: torch.Tensor,
    dtype: torch.dtype,
    shape: Sequence[int],
) -> torch.Tensor:
    torch._check_is_size(blocksize)
    torch._check(
        dtype in [torch.bfloat16, torch.float16, torch.float32],
        lambda: f"Blockwise 4bit dequantization only supports 16/32-bit floats, but got {dtype}",
    )

    # Enable non uint8 dtype
    if A.dtype != torch.uint8:
        A = A.view(torch.uint8)

    A = A.reshape(-1)
    # Map nf4 to [-1, 1]
    out_dq = torch.empty(A.size(0) * 2, dtype=torch.int32, device=A.device)
    n = out_dq.numel()
    out_dq[1::2] = A & 0xF
    out_dq[::2] = A >> 4
    # code is fp32, cast to dtype to avoid the mismatch issue
    code = code.to(dtype).to(A.device)
    out_dq = code[out_dq]

    # Apply scales
    if out_dq.numel() != n:
        assert out_dq.numel() == n + 1
        out_dq = torch.narrow(out_dq, 0, 0, n)
    blocks = n // blocksize
    blocks += 1 if n % blocksize > 0 else 0
    rem = n % blocksize
    has_rem = rem > 0

    out_l = torch.empty(shape, dtype=dtype, device=A.device).reshape(-1)
    if has_rem:
        out_l[: n - rem] = (out_dq[: n - rem].view(-1, blocksize) * absmax[: blocks - has_rem].view(-1, 1)).reshape(-1)
        out_l[n - rem :] = out_dq[n - rem :] * absmax[-1]
    else:
        out_l = out_dq.view(-1, blocksize) * absmax.view(-1, 1)

    out_l = out_l.reshape(-1, *shape[1:]).to(dtype)

    return out_l


@triton.jit
def dequant_4bit_kernel_util(a, offsets, quant_ptr, absmax_ptr, num_paired_elements, QUANT_BLOCK: tl.constexpr):
    PAIRED_QUANT_BLOCK = QUANT_BLOCK // 2
    a = a.to(tl.uint8, bitcast=True)

    # higher 4bits from uint8 packed tensor
    higher = a & 0xF
    # lower 4bits
    lower = a >> 4

    # apply conversion
    higher_nf4 = tl.load(quant_ptr + higher)
    # print("higher : ", higher_nf4)
    lower_nf4 = tl.load(quant_ptr + lower)
    # print("lower uint: ", lower)
    # print("lower     : ", lower_nf4)

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


SMALL_GRF = True


# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': 'small'}, num_warps = 64, num_stages = 2),
#     ],
#     #     triton.Config(
#     #         {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': 'large'},
#     #         num_stages=s, num_warps=32) for s in [1, 2, 3]
#     # ] + [
#     #     triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': m},
#     #                   num_stages=s, num_warps=w)
#     #     for s in [2, 3, 4]
#     #     for (m, w) in ([('large', 32), ('small', 64)] if SMALL_GRF else [('large', 32)])
#     # ] + [
#     #     triton.Config(
#     #         {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': 'large'},
#     #         num_stages=s, num_warps=32) for s in [2]
#     # ] + [
#     #     triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'grf_mode': m},
#     #                   num_stages=s, num_warps=w)
#     #     for s in [2, 3]
#     #     for (m, w) in ([('large', 32), ('small', 64)] if SMALL_GRF else [('large', 32)])
#     # ],
#     key=['M', 'N', 'K'],
# )
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
    stride_ak,
    stride_bn,
    stride_bk,
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
    # offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    # offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_bk = tl.arange(0, BLOCK_SIZE_K // 2)

    # a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_offsets = offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(N, K // 2),
        strides=(stride_bn, stride_bk),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K // 2),
        order=(1, 0),
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # dq_b = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=a_ptr.type.element_ty)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # b_ptrs = b_ptr + b_offsets
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        # print("border: ", K // 2 - k * BLOCK_SIZE_K // 2)
        # print("offs bk: ", offs_bk[None, :])
        # print("mask: ", (offs_bk[None, :] < K // 2 - k * BLOCK_SIZE_K // 2))
        # a = tl.load(a_ptrs, mask=((offs_k[None, :] < K - k * BLOCK_SIZE_K) & (offs_bn[:, None] < N)), other=0.0)
        # b = tl.load(b_ptrs, mask=(offs_bk[None, :] < K // 2 - k * BLOCK_SIZE_K // 2), other=0x77)
        a_blck = tl.load(a_block_ptr, boundary_check=(0, 1))
        b_blck = tl.load(b_block_ptr, boundary_check=(0, 1))
        # print("loaded b: ", b)
        # print("loaded b block offs: ", b_offsets)
        # dq_b_t = dequant_4bit_kernel_util(
        #     b_blck,
        #     b_offsets,
        #     quant_ptr,
        #     absmax_ptr,
        #     num_paired_elements,
        #     QUANT_BLOCK,
        # )
        dq_b_t = dequant_4bit_body_util(a=a_blck, offsets=b_offsets, quant_ptr=quant_ptr, absmax_ptr=absmax_ptr,n_elems= num_paired_elements, QUANT_BLOCK=QUANT_BLOCK)
        # print("dq_b_t: ", dq_b_t)
        dq_b_t = dq_b_t.trans()
        # print(dq_b_t)
        dq_b = dq_b_t.to(a_ptr.type.element_ty)
        # dq_b = dq_b_t

        # We accumulate along the K dimension.
        accumulator += tl.dot(a_blck, dq_b, out_dtype=tl.float32)
        # Advance the ptrs to the next K block.
        # a_ptrs += BLOCK_SIZE_K * stride_ak
        b_offsets += (BLOCK_SIZE_K // 2) * stride_bk
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (0, BLOCK_SIZE_K // 2))
    # c = accumulator.to(a_ptr.type.element_ty)
    c = accumulator.to(c_ptr.type.element_ty)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


def matmul(a, b, shapeB, code, absmax, blocksize):
    # Check constraints.
    # assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    N, K = shapeB

    # b_padded = torch.zeros((N, K//2 + K%2), device=a.device, dtype=torch.uint8)
    # if b.numel() % b_padded.numel() != 0:
    #     b_padded = b_padded.view(-1)
    #     b_padded[:b.numel()] = b.view(-1)
    #     b_padded = b_padded.view(N, K//2 + K%2)
    # b = b_padded
    # print("b shape: ", b.shape, " K + rest: ", K//2 + K%2)
    b = b.view(N, K // 2)
    # b = b.to(torch.uint8)
    # print("b: ", b.shape)
    # print("a.dtype: ",  a.dtype)
    # print("b.dtype: ",  a.dtype)

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M = 16, 16, 16, 4
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    # grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    # print("grid: M - ", triton.cdiv(M, BLOCK_SIZE_M), " N - ", triton.cdiv(N, BLOCK_SIZE_N))
    # print("state code: ", code)
    number_of_paired_elements = b.numel()
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        code,
        absmax,
        number_of_paired_elements,
        blocksize,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,  #
        GROUP_SIZE_M,
    )
    return c


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1})],
    #   + [
    #     triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': m},
    #                   num_stages=s, num_warps=w)
    #     for s in [2, 3, 4]
    #     for (m, w) in ([('large', 32), ('small', 64)] if SMALL_GRF else [('large', 32)])
    # ] + [
    #     triton.Config(
    #         {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': 'large'},
    #         num_stages=s, num_warps=32) for s in [2]
    # ] + [
    #     triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'grf_mode': m},
    #                   num_stages=s, num_warps=w)
    #     for s in [2, 3]
    #     for (m, w) in ([('large', 32), ('small', 64)] if SMALL_GRF else [('large', 32)])
    # ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel_with_block_pointers_nodq(
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
    stride_cn,  #
    ACCUMULATOR_DTYPE: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See the matrix multiplication tutorial for details.
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
    # Create block pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction and accumulate.
    # See above `Make a Block Pointer` section for details.
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0),
    )

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUMULATOR_DTYPE)
    for k in range(0, K, BLOCK_SIZE_K):
        # Load with boundary checks, no need to calculate the mask manually.
        # For better performance, you may remove some axis from the boundary
        # check, if you can guarantee that the access is always in-bound in
        # that axis.
        # See above `Load/Store a Block Pointer` section for details.
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b, out_dtype=ACCUMULATOR_DTYPE)
        # Advance the block pointer to the next K block.
        # See above `Advance a Block Pointer` section for details.
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))
    c = accumulator.to(c_ptr.type.element_ty)
    # ----------------------------------------------------------------
    # Write back the block of the output matrix C with boundary checks.
    # See above `Load/Store a Block Pointer` section for details.
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


def simple_mm(a, b, accum_dtype, res_dtype):
    print("a shape: ", a.shape, " b shape: ", b.shape)
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    # Map accumulator type, e.g. `torch.float16` -> `tl.fp16`
    triton_accum_dtype = tl.dtype(str(accum_dtype)[6:].replace("bfloat", "bf").replace("float", "fp"))
    print("triton accum dtype: ", triton_accum_dtype, " res dtype: ", res_dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    matmul_kernel_with_block_pointers_nodq[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        ACCUMULATOR_DTYPE=triton_accum_dtype,
    )
    return c


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_xpu():
    return triton.runtime.driver.active.get_current_target().backend == "xpu"


def num_sms():
    if is_cuda():
        return torch.cuda.get_device_properties("cuda").multi_processor_count
    if is_xpu():
        return torch.xpu.get_device_properties("xpu").gpu_eu_count
    return 148


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K, WS = args["M"], args["N"], args["K"], args.get("WARP_SPECIALIZE", False)
    ws_str = "_ws" if WS else ""
    ret["name"] = f"{kernel.name}{ws_str} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


def matmul_get_configs(pre_hook=None):
    return [
        triton.Config(
            {"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": 8},
            num_stages=s,
            num_warps=w,
            pre_hook=pre_hook,
        )
        for BM in [128]
        for BN in [128, 256]
        for BK in [64, 128]
        for s in ([3, 4])
        for w in [4, 8]
    ]


@triton.autotune(
    configs=matmul_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    M,
    N,
    K,  #
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,  #
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    # NOTE: There is currently a bug in blackwell pipelining that means it can't handle a value being
    # used in both the prologue and epilogue, so we duplicate the counters as a work-around.
    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if c_ptr.dtype.element_ty == tl.float8e4nv:
            c = accumulator.to(tl.float8e4nv)
        else:
            c = accumulator.to(tl.float16)
        tl.store(c_ptrs, c, mask=c_mask)


def matmul_persistent(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    NUM_SMS = num_sms()

    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])),)
    matmul_kernel_persistent[grid](
        a,
        b,
        c,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        NUM_SMS=NUM_SMS,  #
    )
    return c
