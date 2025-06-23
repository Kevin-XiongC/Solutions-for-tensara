import triton
import triton.language as tl


@triton.jit
def sigmoid_kernel(
    x_ptr: tl.tensor,
    y_ptr: tl.tensor,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n
    x = tl.load(x_ptr + offset, mask=mask)
    y = 1 / (1 + tl.exp(-x))
    tl.store(y_ptr + offset, y, mask=mask)


# Note: input, output are all float32 device tensors
def solution(input, output, n: int, m: int, alpha: float):
    N = n * m
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    sigmoid_kernel[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)
