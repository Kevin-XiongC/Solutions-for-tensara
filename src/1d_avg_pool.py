import triton
import triton.language as tl


@triton.jit
def _average_pooling_kernel(
    input_ptr,
    output_ptr,
    H: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    H_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute output offsets
    pid = tl.program_id(0)
    output_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_mask = output_offsets < H_out

    # Compute starting input positions for each output element
    input_start = output_offsets * stride - padding

    # Initialize accumulator with proper shape
    accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Optimized kernel loop using tl.range for better performance
    for k in tl.range(kernel_size):
        # Compute input indices for all output positions at once
        input_indices = input_start + k

        # Create vectorized mask for boundary conditions
        load_mask = (input_indices >= 0) & (input_indices < H) & output_mask

        # Vectorized load of input values
        input_values = tl.load(input_ptr + input_indices, mask=load_mask, other=0.0)

        # Accumulate values
        accumulator += input_values

    # Compute average and store results
    result = accumulator / kernel_size
    tl.store(output_ptr + output_offsets, result, mask=output_mask)


def solution(input, kernel_size: int, stride: int, padding: int, output, H: int):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    output = torch.empty(H_out).cuda()
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(H_out, meta["BLOCK_SIZE"]),)
    _average_pooling_kernel[grid](
        input,
        output,
        H,
        kernel_size,
        stride,
        padding,
        H_out,
        BLOCK_SIZE,
    )
    return output


import torch

kernel_size = 3
stride = 1
padding = 1
H = 10  # Changed to match input size
input = torch.ones(10).cuda()
output = solution(input, kernel_size, stride, padding, None, H)
print("Our output:", output)
print(
    "PyTorch output:",
    torch.nn.functional.avg_pool1d(
        input.view(1, 1, -1), kernel_size, stride, padding
    ).view(-1),
)
print(
    "Are they equal?",
    torch.allclose(
        torch.nn.functional.avg_pool1d(
            input.view(1, -1), kernel_size, stride, padding
        ).view(-1),
        output,
    ),
)
