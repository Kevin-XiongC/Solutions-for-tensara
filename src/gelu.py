import triton
import triton.language as tl
import torch


@triton.jit
def solution_kernel(input_ptr, alpha, output_ptr, n, m, BLOCK_SIZE: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    off_x = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off_y = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off_z = off_y[:, None] * n + off_x[None, :]

    mask_x = off_x < n
    mask_y = off_y < m
    mask_z = mask_y[:, None] & mask_x[None, :]

    x = tl.load(input_ptr + off_z, mask=mask_z)

    y = tl.where(x < 0, alpha * x, x)
    tl.store(output_ptr + off_z, y, mask=mask_z)


# Note: input, output are all float32 device tensors
def solution(input, alpha: float, output, n: int, m: int):
    BLOCK_SIZE = 128

    # 计算网格大小
    grid = lambda META: (
        triton.cdiv(m, META["BLOCK_SIZE"]),
        triton.cdiv(n, META["BLOCK_SIZE"]),
    )

    solution_kernel[grid](input, alpha, output, n, m, BLOCK_SIZE)


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
