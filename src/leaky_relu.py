import triton
import triton.language as tl
import torch


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


@triton.jit
def optimized_leaky_relu_1d_kernel(
    input_ptr,
    alpha,
    output_ptr,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.where(x >= 0, x, alpha * x)
    tl.store(output_ptr + offsets, y, mask=mask)


# Note: input, output are all float32 device tensors
def solution(input, alpha: float, output, n: int, m: int):
    total_elements = n * m
    if total_elements > 1024 * 1024:
        # Use 1D kernel for large tensors
        BLOCK_SIZE = 1024
        grid = lambda META: (triton.cdiv(total_elements, META["BLOCK_SIZE"]),)

        optimized_leaky_relu_1d_kernel[grid](
            input, alpha, output, total_elements, BLOCK_SIZE
        )
    else:
        # Adaptive block sizing
        if n >= m:
            BLOCK_SIZE_X = min(128, max(16, triton.next_power_of_2(n // 8)))
            BLOCK_SIZE_Y = min(32, max(16, triton.next_power_of_2(m // 4)))
        else:
            BLOCK_SIZE_X = min(32, max(16, triton.next_power_of_2(n // 4)))
            BLOCK_SIZE_Y = min(128, max(16, triton.next_power_of_2(m // 8)))

        grid = lambda META: (
            triton.cdiv(n, META["BLOCK_SIZE_X"]),
            triton.cdiv(m, META["BLOCK_SIZE_Y"]),
        )

        optimized_leaky_relu_kernel[grid](
            input, alpha, output, n, m, BLOCK_SIZE_X, BLOCK_SIZE_Y
        )


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
