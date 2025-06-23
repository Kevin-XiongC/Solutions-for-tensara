import triton
import triton.language as tl


@triton.jit
def _1d_sum_kernel(
    x_ptr,
    y_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_x = offset % K + offset // K * M * K
    mask_x = offset < N * K

    buf = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for i in tl.range(0, tl.cdiv(M, BLOCK_SIZE)):
        offset_y = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask_y = offset_y < M
        input_indices = offset_x[:, None] + offset_y[None, :] * K
        mask = mask_x[:, None] & mask_y[None, :]
        input_values = tl.load(x_ptr + input_indices, mask=mask, other=0)
        buf += tl.sum(input_values, axis=1)

    tl.store(y_ptr + offset, buf, mask=mask_x)


# Note: input, output, shape are all float32 device tensors
def solution(input, dim: int, output, shape, ndim: int):
    import math

    N = math.prod([shape[i].item() for i in range(dim)], start=1)
    M = shape[dim].item()
    K = math.prod([shape[i].item() for i in range(dim + 1, ndim)], start=1)
    BLOCK_SIZE = 256

    grid = lambda meta: (triton.cdiv(N * K, meta["BLOCK_SIZE"]),)
    _1d_sum_kernel[grid](input, output, N, M, K, BLOCK_SIZE)
