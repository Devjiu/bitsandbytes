import torch

import triton
import triton.language as tl

# Should be the same for quant/dequant
from .utils import _FP4_QUANT_TABLE, _NF4_QUANT_TABLE


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
        triton.Config({'SPLIT_SIZE': 512}),
        # triton.Config({'SPLIT_SIZE': 1024}),
    ],
    key=['num_paired_elements', 'QUANT_BLOCK'],
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
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
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
        triton.Config({'SPLIT_SIZE': 512}),
        # triton.Config({'SPLIT_SIZE': 1024}),
    ],
    key=['xnumel', 'QUANT_BLOCK'],
)
@triton.jit
def dequantize_kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, QUANT_BLOCK: tl.constexpr, SPLIT_SIZE: tl.constexpr):
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
        in_ptr0: Pointer to the quantized data (uint8).
        in_ptr1: Pointer to the quantization codebook (float32).
        in_ptr2: Pointer to the absmax values (float32).
        out_ptr0: Pointer to the output tensor (float32).
        xnumel: Number of elements to process.
        XBLOCK: Block size for Triton.
    """
    xoffset = tl.program_id(0) * SPLIT_SIZE
    xindex = xoffset + tl.arange(0, SPLIT_SIZE)
    xmask = xindex < xnumel  # Ensure we don't go out of bounds

    # Load the quantized data
    quantized_indices = tl.load(in_ptr0 + xindex, mask=xmask, other=0).to(tl.int32)

    # Calculate block index and absmax offset
    block_index = xindex // QUANT_BLOCK
    is_valid_block = block_index < (xnumel // QUANT_BLOCK)

    # Load the absmax value for the block
    absmax_offset = tl.where(is_valid_block, block_index, 0)  # Ensure valid offset
    absmax = tl.load(in_ptr2 + absmax_offset, mask=is_valid_block, other=0, eviction_policy='evict_last')

    # Load the dequantized values from the codebook
    # Handle potential out-of-bounds indices
    oob_mask = (quantized_indices < 0) | (quantized_indices >= 256)
    quantized_indices = tl.where(oob_mask, tl.zeros(quantized_indices.shape, tl.int32), quantized_indices)

    dequantized_values = tl.load(in_ptr1 + quantized_indices, mask=xmask, other=0, eviction_policy='evict_last')

    # Scale the dequantized values by absmax
    output_values = dequantized_values * absmax

    # Store the results
    tl.store(out_ptr0 + xindex, output_values, mask=xmask)


def dequant_int8_blockwise(
    A_nf4: torch.Tensor,
    quant_state_code: torch.Tensor,
    absmax: torch.Tensor,
    out: torch.Tensor,
    quant_blocksize: int = 64,
):
    number_of_paired_elements = A_nf4.numel()

    # SPLIT_SIZE = 256
    grid = lambda META: (triton.cdiv(number_of_paired_elements, META["SPLIT_SIZE"]), )
    # grid = (triton.cdiv(number_of_paired_elements, SPLIT_SIZE),)
    dequantize_kernel[grid] (
        A_nf4, quant_state_code, absmax, out, number_of_paired_elements, quant_blocksize,
    )
    # dequant_8bit_kernel[grid](
    #     A_nf4, out, quant_state_code, absmax, number_of_paired_elements, quant_blocksize, # SPLIT_SIZE
    # )
    return out

@torch.compile
def dequantize_8bit_blockwise_torch(
    A_nf4: torch.Tensor,
    quant_state_code: torch.Tensor,
    absmax: torch.Tensor,
    out: torch.Tensor,
    quant_blocksize: int = 64,
):
    """
    Dequantizes an int8 blockwise quantized tensor using torch operations.

    Args:
        A_nf4: The int8 quantized tensor.
        quant_state_code: The quantization codebook (usually _NF4_QUANT_TABLE or _FP4_QUANT_TABLE).
        absmax: The absolute maximum values for each block.
        quant_blocksize: The block size used for quantization.

    Returns:
        The dequantized tensor.
    """

    # n = A_nf4.numel()
    out_shape = A_nf4.shape  # Preserve the original shape
    # A_nf4 = A_nf4.flatten()  # Flatten for easier indexing

    # Calculate the number of blocks
    out = quant_state_code[A_nf4.reshape(-1).int()]
    blocks = out.shape[-1] // quant_blocksize
    res = out.shape[-1] % quant_blocksize
    if res != 0:
        out = torch.nn.functional.pad(out, (0, quant_blocksize - res), mode="constant", value=0)
    out = (out.view(-1, quant_blocksize) * absmax.view(-1, 1)).to(out.dtype).reshape(-1)
    out = out[: blocks * quant_blocksize + res]
    out = out.reshape(A_nf4.shape)

    return out.reshape(out_shape)

@triton.autotune(
    configs=[
        triton.Config({'SPLIT_NUM_BLOCKS': 1}),
        # triton.Config({'SPLIT_NUM_BLOCKS': 1, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
        # triton.Config({'SPLIT_NUM_BLOCKS': 1, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
        # triton.Config({'SPLIT_NUM_BLOCKS': 1, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
        # #
        # triton.Config({"SPLIT_NUM_BLOCKS": 1, "grf_mode": "auto"}, num_stages=4, num_warps=32),
        # #
        # triton.Config({"SPLIT_NUM_BLOCKS": 2, "grf_mode": "large"}, num_stages=2, num_warps=32),
        # # triton.Config({'SPLIT_NUM_BLOCKS': 2, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
        # triton.Config({"SPLIT_NUM_BLOCKS": 2, "grf_mode": "auto"}, num_stages=2, num_warps=32),
        # triton.Config({"SPLIT_NUM_BLOCKS": 2, "grf_mode": "auto"}, num_stages=4, num_warps=32),
        # triton.Config({"SPLIT_NUM_BLOCKS": 4, "grf_mode": "large"}, num_stages=2, num_warps=32),
        # triton.Config({"SPLIT_NUM_BLOCKS": 4, "grf_mode": "large"}, num_stages=4, num_warps=32),
        # triton.Config({'SPLIT_NUM_BLOCKS': 8, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
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

    # This can be fruitful, but compiler should preload it
    # code = tl.load(code_ptr + tl.arange(0, CODE_SIZE))

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


@triton.jit
def unite_2_int4(x, y):
    return (x & 0xF) | (y << 4)


@triton.autotune(
    configs=[
        # triton.Config({'SPLIT_NUM_BLOCKS': 1, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
        # triton.Config({'SPLIT_NUM_BLOCKS': 1, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
        # triton.Config({'SPLIT_NUM_BLOCKS': 1, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
        # #
        # triton.Config({"SPLIT_NUM_BLOCKS": 1, "grf_mode": "auto"}, num_stages=4, num_warps=32),
        #
        triton.Config({"SPLIT_NUM_BLOCKS": 2}),
        # triton.Config({"SPLIT_NUM_BLOCKS": 2, "grf_mode": "large"}, num_stages=2, num_warps=32),
        # # triton.Config({'SPLIT_NUM_BLOCKS': 2, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
        # triton.Config({"SPLIT_NUM_BLOCKS": 2, "grf_mode": "auto"}, num_stages=2, num_warps=32),
        # triton.Config({"SPLIT_NUM_BLOCKS": 2, "grf_mode": "auto"}, num_stages=4, num_warps=32),
        # triton.Config({"SPLIT_NUM_BLOCKS": 4, "grf_mode": "large"}, num_stages=2, num_warps=32),
        # triton.Config({"SPLIT_NUM_BLOCKS": 4, "grf_mode": "large"}, num_stages=4, num_warps=32),
        # triton.Config({'SPLIT_NUM_BLOCKS': 8, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
    ],
    key=["n_elements", "BLOCK_SIZE"],
)
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

    packed = tl.reduce(quantized, axis=2, combine_fn=unite_2_int4)
    packed = packed.to(tl.uint8, bitcast=True)

    packed_flat = tl.reshape(packed, (BLOCK_SIZE * SPLIT_NUM_BLOCKS,))
    out_offsets = block_start_idx * BLOCK_SIZE // 2 + tl.arange(0, SPLIT_NUM_BLOCKS * BLOCK_SIZE)
    out_mask = out_offsets < n_elements // 2
    tl.store(out_ptr + out_offsets, packed_flat, mask=out_mask)


def quantize_4bit_blockwise_triton(A, blocksize, code, blocks, absmax, quantized_out):
    n = A.numel()

    # split_num_blocks = 1
    # grid = (triton.cdiv(blocks, split_num_blocks),)
    # grid = (1, )
    grid = lambda META: (triton.cdiv(blocks, META["SPLIT_NUM_BLOCKS"]),)
    # print(" blocksize, split_num_blocks: ", blocksize, split_num_blocks)
    # print(" blocksize, split_num_blocks: ", blocksize, split_num_blocks*2)
    # print("A shape: ", A.shape, " numel: ", n, " blocks: ", blocks)
    quantize_4bit_blockwise_kernel[grid](
        A_ptr=A,
        code_ptr=code,
        absmax_ptr=absmax,
        out_ptr=quantized_out,
        n_elements=n,
        BLOCK_SIZE=blocksize,
        CODE_SIZE=code.numel(),
        # SPLIT_NUM_BLOCKS=split_num_blocks * 2,
    )

    return quantized_out, absmax


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
        triton.Config({'SPLIT_SIZE': 512}),
        # triton.Config({'SPLIT_SIZE': 512, 'grf_mode': 'large'}, num_stages=2, num_warps=32),
        # triton.Config({'SPLIT_SIZE': 512, 'grf_mode': 'auto'}, num_stages=2, num_warps=32),
        # triton.Config({'SPLIT_SIZE': 512, 'grf_mode': 'large'}, num_stages=4, num_warps=32),
        # triton.Config({'SPLIT_SIZE': 512, 'grf_mode': 'auto'}, num_stages=4, num_warps=32),
        # triton.Config({'SPLIT_SIZE': 1024}),
        # triton.Config({'SPLIT_SIZE': 2048}),
        # triton.Config({'SPLIT_SIZE': 4096}),
        # triton.Config({'SPLIT_SIZE': 8192}),
        # triton.Config({'SPLIT_SIZE': 16384}),
    ],
    key=['num_paired_elements', 'QUANT_BLOCK'],
)
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

    # SPLIT_SIZE = 512
    grid = lambda META: (triton.cdiv(number_of_paired_elements, META['SPLIT_SIZE']), )
    # grid = (triton.cdiv(number_of_paired_elements, SPLIT_SIZE),)
    if quant_type == "fp4":
        dequant_4bit_kernel[grid](A, out, _FP4_QUANT_TABLE, absmax, number_of_paired_elements, blocksize)#, SPLIT_SIZE)
    else:
        dequant_4bit_kernel[grid](A, out, _NF4_QUANT_TABLE, absmax, number_of_paired_elements, blocksize)#, SPLIT_SIZE)


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

    SPLIT_SIZE = 512
    # grid = lambda META: (triton.cdiv(number_of_paired_elements, SPLIT_SIZE), )
    grid = (triton.cdiv(number_of_paired_elements, SPLIT_SIZE),)
    dequant_4bit_kernel[grid](A, out, code, absmax, number_of_paired_elements, blocksize, SPLIT_SIZE)
