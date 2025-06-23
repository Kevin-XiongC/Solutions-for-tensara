import triton
import triton.language as tl


@triton.jit
def optimized_leaky_relu_kernel(
    input_ptr,
    alpha,
    output_ptr,
    n,
    m,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    start_x = pid_x * BLOCK_SIZE_X
    start_y = pid_y * BLOCK_SIZE_Y

    off_x = start_x + tl.arange(0, BLOCK_SIZE_X)
    off_y = start_y + tl.arange(0, BLOCK_SIZE_Y)
    off_2d = off_y[:, None] * n + off_x[None, :]

    mask_x = off_x < n
    mask_y = off_y < m
    mask_2d = mask_y[:, None] & mask_x[None, :]

    x = tl.load(input_ptr + off_2d, mask=mask_2d, other=0.0)
    y = tl.where(x >= 0, x, alpha * x)
    tl.store(output_ptr + off_2d, y, mask=mask_2d)


import torch

if __name__ == "__main__":
    m, n = (2857, 1718)
    input = torch.randn(m, n, device="cuda")
    alpha = 0.1
    output = torch.zeros_like(input, device="cuda")
    solution(input, alpha, output, n, m)

    print(
        torch.allclose(
            output, torch.nn.functional.leaky_relu(input, alpha, inplace=True)
        )
    )
