import triton
import triton.language as tl


@triton.jit
def cos_loss_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    n: tl.constexpr,
    d: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    pid_x = tl.program_id(0)
    offset_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    mask_x = offset_x < n

    reduce_buffer = tl.zeros([BLOCK_SIZE_X], dtype=tl.float32)
    x_norm = tl.zeros([BLOCK_SIZE_X], dtype=tl.float32)
    y_norm = tl.zeros([BLOCK_SIZE_X], dtype=tl.float32)

    for i in tl.range(0, tl.cdiv(d, BLOCK_SIZE_Y), num_stages=3):
        offset_y = i * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
        mask_y = offset_y < d
        offset_xy = offset_x[:, None] * d + offset_y[None, :]  # Correct 2D indexing
        mask_xy = mask_x[:, None] & mask_y[None, :]

        predictions = tl.load(predictions_ptr + offset_xy, mask=mask_xy)
        targets = tl.load(targets_ptr + offset_xy, mask=mask_xy)

        reduce_buffer += tl.sum(predictions * targets, axis=1)
        x_norm += tl.sum(predictions * predictions, axis=1)
        y_norm += tl.sum(targets * targets, axis=1)

    x_norm = tl.maximum(tl.sqrt(x_norm), 1e-8)
    y_norm = tl.maximum(tl.sqrt(y_norm), 1e-8)
    cos_loss = reduce_buffer / (x_norm * y_norm)
    tl.store(output_ptr + offset_x, cos_loss, mask=mask_x)


def solution(predictions, targets, output, n: int, d: int):
    BLOCK_SIZE_X = 256
    BLOCK_SIZE_Y = 64
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_X"]),)
    cos_loss_kernel[grid](
        predictions, targets, output, n, d, BLOCK_SIZE_X, BLOCK_SIZE_Y
    )


# Test with sample data
import torch

n, d = 3, 4
predictions = torch.rand((n, d), device="cuda", dtype=torch.float32)
targets = torch.rand((n, d), device="cuda", dtype=torch.float32)
output = torch.zeros(n, device="cuda", dtype=torch.float32)

solution(predictions, targets, output, n, d)
ref = torch.nn.functional.cosine_similarity(predictions, targets, dim=1)

print("Triton output:", output.mean())
print("Reference:", ref.mean())
