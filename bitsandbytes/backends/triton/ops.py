from collections.abc import Sequence

import torch

from bitsandbytes.functional import get_4bit_type

_NF4_QUANT_TABLE = get_4bit_type("nf4", device="xpu")

try:
    from . import triton_kernels

    triton_available = True
except ImportError as e:
    print("Import error:", e)
    triton_available = False

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
    torch._check(n % blocksize == 0, lambda: f"n must be divisible by blocksize, got {n} and {blocksize}")

    blocks = -(n // -(blocksize * 2))

    absmax = torch.empty((blocks * 2,), device=A.device, dtype=A.dtype)
    out = torch.empty((n // 2, 1), device=A.device, dtype=torch.uint8)

    if quant_type == "fp4":
        triton_kernels.quantize_fp4_blockwise_triton(A, blocksize, blocks, absmax, out)
    else:
        triton_kernels.quantize_nf4_blockwise_triton(A, blocksize, blocks, absmax, out)
        # triton_kernels.quantize_nf4_blockwise_triton(A, blocksize, _NF4_QUANT_TABLE, blocks, absmax, out)
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

    # Check if this is fine and fast
    if A.dtype != torch.uint8:
        A = A.squeeze().view(torch.uint8).unsqueeze(1)

    out = torch.empty(shape, dtype=dtype, device=A.device)

    triton_kernels._dequantize_4bit_impl(A, absmax, blocksize, quant_type, dtype, out=out)
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

    # B_dq = torch.empty(shapeB, dtype=A.dtype, device=A.device)
    # triton_kernels._dequantize_4bit_impl_passing_code(
    #     B,
    #     absmax,
    #     blocksize,
    #     code,
    #     dtype=A.dtype,
    #     out=B_dq,
    # )
    # print("B_dq: ", B_dq.shape)
    # print("B_dq: ", B_dq)
    return triton_kernels.matmul(A, B, shapeB, code, absmax, blocksize)
    # quant_type = "fp4" if code[1] > 0 else "nf4"
    # B_dq = dequantize_4bit(B, absmax, blocksize, quant_type, shapeB, A.dtype)

    # For some reason directly passing code causes errors in some cases like:
    # tests/test_functional.py::TestQuantize4BitFunctional::test_gemv_4bit[dim=128-uint8-fp32-fc1-fp4-DQ_True-xpu]
    #
    B_dq_triton = torch.empty(shapeB, dtype=A.dtype, device=A.device)
    # B_dq = B_dq * 7
    # code = code.to(A.device).sort().values
    # if B.dtype != torch.uint8:
    #     B = B.squeeze().view(torch.uint8).unsqueeze(1)

    # triton_kernels._dequantize_4bit_impl_passing_code(
    #     B,
    #     absmax,
    #     blocksize,
    #     code,
    #     dtype=A.dtype,
    #     out=B_dq_triton,
    # )

    B_dq = triton_kernels.dequant_4bit_blockwise(B,
        absmax,
        blocksize,
        code,
        dtype=A.dtype,
        shape=shapeB,
    )
    # print("allclose: ", torch.allclose(B_dq_triton, B_dq))
    # print("absmax diff: ", torch.max((B_dq_triton - B_dq).abs()))

    # User called gemv with B.t(), so we need to transpose it back.
    # if B.shape[0] == 1:
    #    B_dq = B_dq.t()
    # A = A.to(torch.float32)
    # B_pass = B_dq.to(torch.float32)
    # c_ref = torch.matmul(
    #     A,
    #     B_dq.T,
    # )
    # print("c_ref: ", c_ref.dtype)
    # print("c_ref: ", c_ref.shape)

    # print("B_d before passing to triton: ", B_pass.shape)
    # print("B_d before passing to triton: ", B_pass)
    # print("A: ", A)
    # print("c_ref: ", c_ref)
    print("B_dq shape: ", B_dq.shape)
    # c = triton_kernels.simple_mm(
    #     A.to(torch.float32),
    #     B_dq.T.to(torch.float32),
    #     A.dtype,
    #     A.dtype,
    # )

    # ones = torch.eye(c.shape[1], c.shape[1], device=c.device, dtype=c.dtype)
    # c = torch.matmul(c, ones)
    # import pdb
    # pdb.set_trace()
    # print(" c: ", c[0][0])
    # c = c.to(A.dtype)
    # torch.set_float32_matmul_precision("high")
    # c_mm_ref = torch.matmul(A, B_dq_triton.T)
    # print("diff: ", c_mm_ref - c)
    # print("c: ", c)
    # print("c_ref == c: ", c_ref - c)

    # errs1 = (c - c_ref).abs().float()
    # errs1.append(err1.mean().item())
    # err1 = sum(errs1) / len(errs1) / math.sqrt(dim)
    # torch.set_float32_matmul_precision("highest")
    out_mm_tri = torch.nn.functional.linear(
        A,
        B_dq,
        bias=None,
    )

    out_mm_torch = torch.nn.functional.linear(
        A,
        B_dq,
        bias=None,
    )
    print("out allclose: ", torch.allclose(out_mm_torch, out_mm_tri))
    print("out absmax diff: ", torch.max((out_mm_torch - out_mm_tri).abs()))
    print("")

    return out_mm_tri.xpu()

    # return c_mm_ref
