import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


# Note: d_input1, d_input2, d_output are all float32 device tensors
def solution(d_input1, d_input2, d_output, n: int):
    BLOCK_SIZE_X = 256
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_X"]),)
    add_kernel[grid](d_input1, d_input2, d_output, n, BLOCK_SIZE_X)


import torch

n = 1024
d_input1 = torch.rand((n,), device="cuda", dtype=torch.float32)
d_input2 = torch.rand((n,), device="cuda", dtype=torch.float32)
d_output = torch.empty((n,), device="cuda", dtype=torch.float32)

solution(d_input1, d_input2, d_output, n)
