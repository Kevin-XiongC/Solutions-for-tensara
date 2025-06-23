import triton
import triton.language as tl


@triton.jit
def _max_pooling_kernel(
    input_ptr,
    output_ptr,
    H: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    H_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    dilation: tl.constexpr,
):
    # Get program ID and compute output offsets
    pid = tl.program_id(0)
    output_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_mask = output_offsets < H_out

    # Compute starting input positions for each output element
    input_start = output_offsets * stride - padding

    # Initialize accumulator with proper shape
    accumulator = tl.full([BLOCK_SIZE], -float("inf"), dtype=tl.float32)

    # Optimized kernel loop using tl.range for better performance
    for k in tl.range(0, kernel_size, num_stages=3):
        # Compute input indices for all output positions at once
        input_indices = input_start + k * dilation

        # Create vectorized mask for boundary conditions
        load_mask = (input_indices >= 0) & (input_indices < H)

        # Vectorized load of input values
        input_values = tl.load(
            input_ptr + input_indices, mask=load_mask, other=float("-inf")
        )

        # Accumulate values
        accumulator = tl.maximum(accumulator, input_values)

    # Compute average and store results
    result = accumulator
    tl.store(output_ptr + output_offsets, result, mask=output_mask)


def solution(
    input, kernel_size: int, stride: int, padding: int, dilation: int, output, H: int
):
    H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(H_out, meta["BLOCK_SIZE"]),)
    _max_pooling_kernel[grid](
        input,
        output,
        H,
        kernel_size,
        stride,
        padding,
        H_out,
        BLOCK_SIZE,
        dilation,
    )


import torch

torch.manual_seed(0)
kernel_size = 3
stride = 1
padding = 1
H = 14  # Changed to match input size
dilation = 2
# write a UT for the solution considering dilation
input = torch.randn(H).cuda()
H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
output = torch.zeros(H_out).cuda()

solution(input, kernel_size, stride, padding, dilation, output, H)
print(output)
print(
    torch.nn.functional.max_pool1d(
        input.view(1, -1), kernel_size, stride, padding, dilation
    ).view(-1)
)
print(
    torch.allclose(
        output,
        torch.nn.functional.max_pool1d(
            input.view(1, -1), kernel_size, stride, padding, dilation
        ).view(-1),
    )
)
