import triton
import triton.language as tl


@triton.jit
def argmax_kernel(
    input_ptr: tl.tensor,
    output_ptr: tl.tensor,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
):
    pid = tl.program_id(0)
    offset_x = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_x = offset_x < N * K
    offset_x_1 = offset_x % K + offset_x // K * M * K
    ids = tl.full([BLOCK_SIZE], -1, dtype=tl.int32)
    max_val = tl.full([BLOCK_SIZE], -float("inf"), dtype=tl.float32)

    for i in tl.range(0, tl.cdiv(M, BLOCK_SIZE)):
        offset_y = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask_y = offset_y < M
        input_indices = offset_x_1[:, None] + offset_y[None, :] * K
        mask = mask_x[:, None] & mask_y[None, :]

        input_values = tl.load(
            input_ptr + input_indices, mask=mask, other=-float("inf")
        )

        current_max_id = tl.argmax(input_values, axis=1)
        current_max = tl.max(input_values, axis=1)
        is_better = current_max > max_val
        max_val = tl.where(is_better, current_max, max_val)
        ids = tl.where(is_better, current_max_id + i * BLOCK_SIZE, ids)

    tl.store(output_ptr + offset_x, ids, mask=mask_x)


def solution(input, dim: int, output, shape, ndim: int):
    if dim < 0:
        dim += ndim
    assert 0 <= dim < ndim, "dim out of range"
    import math

    N = math.prod([shape[i].item() for i in range(dim)], start=1)
    K = math.prod([shape[i].item() for i in range(dim + 1, ndim)], start=1)
    M = shape[dim].item()
    BLOCK_SIZE = 8
    grid = lambda meta: (triton.cdiv(N * K, meta["BLOCK_SIZE"]),)

    argmax_kernel[grid](input, output, M, N, BLOCK_SIZE, K)


import torch
import numpy as np

output_1 = torch.empty([2, 4], device="cuda", dtype=torch.float32)


arr = np.array(
    [
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
    ]
)
torch.manual_seed(0)
input = torch.tensor(arr, device="cuda", dtype=torch.float32)  # 三维张量
print("Input:\n", input)
output = torch.empty([2, 4], device="cuda", dtype=torch.int32)
solution(input, dim=1, output=output, shape=torch.tensor(input.shape), ndim=input.ndim)
print(torch.argmax(input, dim=1))
print(torch.max(input, dim=1).values)
print(output)
print(output_1)
# for dim in [0, 1, 2, -1]:
#     expected = torch.argmax(input, dim=dim)
#     actual = solution(input, dim=dim)
#     assert torch.allclose(expected, actual), f"dim={dim} failed"
#     print(f"dim={dim}:\nExpected: {expected}\nActual: {actual}")
