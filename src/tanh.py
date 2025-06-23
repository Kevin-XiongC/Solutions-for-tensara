import triton
import triton.language as tl


@triton.jit
def tanh_kernel(x_ptr, y_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n

    x = tl.load(x_ptr + offset, mask=mask, other=0)
    y = tl.exp(2 * x)
    tl.store(y_ptr + offset, 1 - 2 / (1 + y), mask=mask)


# Note: input, output are all float32 device tensors
def solution(input, output, n: int, m: int):
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n * m, meta["BLOCK_SIZE"]),)
    tanh_kernel[grid](input, output, n * m, BLOCK_SIZE)
