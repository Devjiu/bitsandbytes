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
    [ 0.00000,  0.00521,  0.66667,  1.00000,  0.33333,  0.50000,  0.16667,  0.25000,  0.00000, -0.00521, -0.66667,
        -1.00000, -0.33333, -0.50000, -0.16667, -0.25000],
    dtype=torch.float32,
    device="xpu",
)

def print_tensor_bin(tensor):
    arr = tensor.flatten().cpu().numpy()
    for i in range(0, len(arr), 4):
        line = "  ".join(f"{int(v):08b}" for v in arr[i:i+4])
        print(line)

@torch.compile
def quantize_4bit_torch(
    A: torch.Tensor, blocksize: int, quant_type: str, quant_storage: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    # Divide into blocks and normalize
    blocks = A.reshape(-1, blocksize)
    absmax = blocks.abs().max(dim=1).values.float()
    scaled = blocks / absmax.unsqueeze(-1)
    if quant_type == "fp4":
        quantized = torch.argmin(torch.abs(scaled.view(-1, 1) - _FP4_QUANT_TABLE), dim=-1, keepdim=True).to(
            torch.uint8
        )
    else:
        quantized = torch.argmin(torch.abs(scaled.view(-1, 1) - _NF4_QUANT_TABLE), dim=-1, keepdim=True).to(
            torch.uint8
        )
    # print("\ntorch quantized all: ", quantized.flatten())
    # print("\ntorch quantized even: ", quantized[::2].flatten())
    # print("\ntorch quantized even bin format: ", print_tensor_bin(quantized[::2]))
    # print("\ntorch quantized  odd: ", quantized[1::2].flatten())
    # print("\ntorch quantized  odd bin format: ", print_tensor_bin(quantized[1::2]))
    packed = quantized[::2] << 4 | quantized[1::2]
    if quant_storage != torch.uint8:
        packed = packed.squeeze().view(quant_storage).unsqueeze(1)
    return packed, absmax.float()

if triton_available:
    register_kernel("bitsandbytes::quantize_blockwise", "xpu")(triton_ops.quantize_blockwise)
    register_kernel("bitsandbytes::dequantize_blockwise.out", "xpu")(triton_ops.dequantize_blockwise_inplace)
    register_kernel("bitsandbytes::dequantize_blockwise", "xpu")(triton_ops.dequantize_blockwise)
    register_kernel("bitsandbytes::quantize_4bit", "xpu")(triton_ops.quantize_4bit)
    register_kernel("bitsandbytes::dequantize_4bit.out", "xpu")(triton_ops.dequantize_4bit_inplace)
    register_kernel("bitsandbytes::dequantize_4bit", "xpu")(triton_ops.dequantize_4bit)
    register_kernel("bitsandbytes::gemv_4bit", "xpu")(triton_ops.gemv_4bit)
else:
    warnings.warn("XPU available, but trtion package is missing.")
