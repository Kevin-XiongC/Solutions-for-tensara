import triton
import triton.language as tl


@triton.jit
def gemv_kernel(
    A_ptr: tl.tensor,
    B_ptr: tl.tensor,
    C_ptr: tl.tensor,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # C=B*A
    pid = tl.program_id(0)
    offset_x = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_x = offset_x < M
    buf = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for i in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offset_y = i * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_y = offset_y < K

        idx_a = offset_x[:, None] * K + offset_y[None, :]
        idx_b = offset_y

        mask_a = mask_x[:, None] & mask_y[None, :]
        a = tl.load(A_ptr + idx_a, mask=mask_a, other=0)  # (b,b)
        b = tl.load(B_ptr + idx_b, mask=mask_y, other=0)
        t = tl.sum(a * b, axis=1)
        buf += t

    tl.store(C_ptr + offset_x, buf, mask=mask_x)


# Note: input_a, input_b, output_c are all float32 device tensors
def solution(input_a, input_b, output_c, m: int, k: int):
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE"]),)
    gemv_kernel[grid](input_a, input_b, output_c, m, k, BLOCK_SIZE, 64)


# Note: input_a, input_b, output_c are all float32 device tensors


import torch
import numpy as np

torch.manual_seed(0)
device = torch.device("cpu")

M = 4096
K = 4096
input_a = torch.randn(M, K, device=device, dtype=torch.float32)
input_b = torch.randn(K, device=device, dtype=torch.float32)
output_c = torch.empty([M], device=device, dtype=torch.float32)
solution(input_a, input_b, output_c, M, K)
print(torch.mean(torch.abs(output_c - input_a @ input_b)))
print(torch.allclose(output_c, input_a @ input_b))
