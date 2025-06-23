import triton
import triton.language as tl


@triton.jit
def _1d_conv_kernel(
    input_ptr,
    output_ptr,
    H: tl.constexpr,
    kernel_size: tl.constexpr,
    padding: tl.constexpr,
    H_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    kernel_ptr: tl.tensor,
):
    # Get program ID and compute output offsets
    pid = tl.program_id(0)
    output_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_mask = output_offsets < H_out

    # Compute starting input positions for each output element
    input_start = output_offsets - padding

    # Initialize accumulator with proper shape
    accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Optimized kernel loop using tl.range for better performance
    for k in tl.range(0, kernel_size, num_stages=3):
        # Compute input indices for all output positions at once
        input_indices = input_start + k

        # Create vectorized mask for boundary conditions
        load_mask = (input_indices >= 0) & (input_indices < H)

        # Vectorized load of input values
        input_values = tl.load(input_ptr + input_indices, mask=load_mask, other=0)
        kernel = tl.load(kernel_ptr + k)

        accumulator += kernel * input_values

    # Compute average and store results
    result = accumulator
    tl.store(output_ptr + output_offsets, result, mask=output_mask)


def solution(A, B, C, N: int, K: int):
    padding = K // 2
    H_out = N + 2 * padding - K + 1
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(H_out, meta["BLOCK_SIZE"]),)
    input = A
    output = C
    kernel = B
    _1d_conv_kernel[grid](
        input,
        output,
        N,
        K,
        padding,
        H_out,
        BLOCK_SIZE,
        kernel,
    )


import torch


torch.manual_seed(0)
kernel_size = 3
kernel = torch.ones(kernel_size).cuda()
input = torch.ones(14).cuda()
output = torch.zeros(14).cuda()
solution(input, kernel, output, 14, 3)
print(output)
ref = torch.nn.functional.conv1d(
    torch.nn.functional.pad(input.view(1, -1), (1, 1)), kernel.view(1, 1, -1)
).view(-1)
print(ref)
print(torch.allclose(output, ref))
