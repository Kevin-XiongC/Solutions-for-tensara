import triton
import triton.language as tl

from triton.language.math import rsqrt


@triton.jit
def rms_norm_kernel(
    X_ptr,
    Y_ptr,
    B,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    X_ptr += pid * N
    Y_ptr += pid * N
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    X_local = tl.load(X_ptr + offset, mask=mask)

    mean_square = tl.sum(X_local * X_local, axis=0) / N
    rstd = rsqrt(mean_square + 1e-5)

    Y = X_local * rstd
    tl.store(Y_ptr + offset, Y, mask=mask)


# Note: X, Y are all float32 device tensors
def solution(X, Y, B: int, N: int):
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (B,)  # Always launch B programs, one per row

    rms_norm_kernel[grid](X, Y, B, N, BLOCK_SIZE)

import torch

if __name__ == "__main__":
    m, n = (3, 3)
    input = torch.randn(m, n, device="cuda")
    output = torch.zeros_like(input, device="cuda")
    solution(input, output, m, n)
    print(output)
    print(torch.nn.functional.rms_norm(input, (n,)))

