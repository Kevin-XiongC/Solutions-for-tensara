import triton
import triton.language as tl


@triton.jit
def l1_norm_kernel(
    X_ptr,
    Y_ptr,
    B: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    x_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    x_mask = x_offset < B

    for i in tl.range(0, tl.cdiv(D, BLOCK_SIZE), num_stages=3):
        offset_y = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask_y = offset_y < D
        offset_xy = x_offset[:, None] * D + offset_y[None, :]
        mask_xy = x_mask[:, None] & mask_y[None, :]
        X = tl.load(X_ptr + offset_xy, mask=mask_xy)
        acc += tl.sum(tl.abs(X), axis=1)
    
    acc += 1e-10

    for i in tl.range(0, tl.cdiv(D, BLOCK_SIZE), num_stages=3):
        offset_y = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask_y = offset_y < D
        offset_xy = x_offset[:, None] * D + offset_y[None, :]
        mask_xy = x_mask[:, None] & mask_y[None, :]
        X = tl.load(X_ptr + offset_xy, mask=mask_xy)
        tl.store(Y_ptr + offset_xy, X / acc[:, None], mask=mask_xy)


# Note: X, Y are all float32 device tensors
def solution(X, Y, B: int, D: int):
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(B, meta["BLOCK_SIZE"]),)
    l1_norm_kernel[grid](X, Y, B, D, BLOCK_SIZE)
