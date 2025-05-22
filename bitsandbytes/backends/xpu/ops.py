import warnings

from ..._ops import register_kernel

try:
    from ..triton import ops as triton_ops

    triton_available = True
except ImportError as e:
    print("Import error:", e)
    triton_available = False

import torch

_NF4_QUANT_TABLE = torch.tensor(
    [
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
    ],
    dtype=torch.float32,
    device="xpu",
)

# Should be sorted to use binary search
_FP4_QUANT_TABLE = torch.tensor(
    [
        0.00000,
        0.00521,
        0.66667,
        1.00000,
        0.33333,
        0.50000,
        0.16667,
        0.25000,
        0.00000,
        -0.00521,
        -0.66667,
        -1.00000,
        -0.33333,
        -0.50000,
        -0.16667,
        -0.25000,
    ],
    dtype=torch.float32,
    device="xpu",
)


def print_tensor_bin(tensor):
    arr = tensor.flatten().cpu().numpy()
    for i in range(0, len(arr), 4):
        line = "  ".join(f"{int(v):08b}" for v in arr[i : i + 4])
        print(line)


@torch.compile
def quantize_blockwise_torch(A, code, blocksize):
    n = A.numel()
    blocks = -(n // -blocksize)

    absmax = torch.empty((blocks,), device=A.device, dtype=A.dtype)
    quantized_out = torch.empty_like(A.flatten(), dtype=torch.uint8)

    rem = n % blocksize
    has_rem = rem > 0
    blocks = n // blocksize + has_rem
    A_reshaped = A.reshape(n)
    A_com = A_reshaped[: n - rem]
    A_com_reshaped = A_com.reshape(n // blocksize, blocksize)
    absmax[: blocks - has_rem] = torch.abs(A_com_reshaped).max(dim=-1)[0]
    scaled_A = torch.clamp(A_com_reshaped / absmax[: blocks - has_rem].view(-1, 1), -1, 1)
    scaled_A = scaled_A.reshape(-1)
    if has_rem:
        absmax[-1] = torch.abs(A_reshaped[n - rem :]).max()
        scaled_A_rem = torch.clamp((A_reshaped[n - rem :] / absmax[-1]), -1, 1)
        scaled_A = torch.cat([scaled_A, scaled_A_rem], dim=0)

    diff = torch.abs(scaled_A.unsqueeze(-1) - code.to(scaled_A.device))
    quantized_out = torch.argmin(diff, dim=-1).to(torch.uint8).to(scaled_A.device).reshape(A.shape)
    return quantized_out, absmax


if triton_available:
    register_kernel("bitsandbytes::quantize_blockwise", "xpu")(triton_ops.quantize_blockwise)
    # register_kernel("bitsandbytes::quantize_blockwise", "xpu")(quantize_blockwise_torch)
    register_kernel("bitsandbytes::dequantize_blockwise.out", "xpu")(triton_ops.dequantize_blockwise_inplace)
    register_kernel("bitsandbytes::dequantize_blockwise", "xpu")(triton_ops.dequantize_blockwise)
    register_kernel("bitsandbytes::quantize_4bit", "xpu")(triton_ops.quantize_4bit)
    register_kernel("bitsandbytes::dequantize_4bit.out", "xpu")(triton_ops.dequantize_4bit_inplace)
    register_kernel("bitsandbytes::dequantize_4bit", "xpu")(triton_ops.dequantize_4bit)
    register_kernel("bitsandbytes::gemv_4bit", "xpu")(triton_ops.gemv_4bit)
    # register_kernel("bitsandbytes::gemv_4bit.out", "xpu")(triton_ops.gemv_4bit_inpalce)
else:
    warnings.warn("XPU available, but trtion package is missing.")
