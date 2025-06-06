import torch
import triton
import time
import numpy as np
import triton.language as tl
import cupy as cp

"""CPU Implementation"""
def distance_dot_cpu(X, Y):
    # For dot product as a distance, we use negative dot product
    # since similar vectors have larger dot products
    return float(-np.dot(X, Y))


""" Cupy Implementation"""
def distance_dot_cupy(X, Y):
    if not isinstance(X, cp.ndarray):
        X = cp.asarray(X, dtype=cp.float32)
    if not isinstance(Y, cp.ndarray):
        Y = cp.asarray(Y, dtype=cp.float32)

    # Return negative dot product as distance
    return float(-cp.dot(X, Y))

"""Atomic-Add implementation in Triton"""
@triton.jit
def dot_kernel_atomic(
    x_ptr, y_ptr, dot_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Create block pointers
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute partial dot product for this block
    partial_dot = tl.sum(x * y)
    
    # Atomic add to the shared result
    tl.atomic_add(dot_ptr, partial_dot)

def distance_dot_atomic(X, Y):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, device='cuda', dtype=torch.float32)
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y, device='cuda', dtype=torch.float32)
    
    n_elements = X.numel()
    
    # Create intermediate buffer for dot product
    dot_product = torch.zeros(1, device='cuda', dtype=torch.float32)
    
    # Determine block size (power of 2)
    block_size = triton.next_power_of_2(min(n_elements, 1024))
    
    # Compute grid size - how many blocks we need
    n_blocks = triton.cdiv(n_elements, block_size)
    grid = (n_blocks,)
    
    # Launch kernel to compute partial results
    dot_kernel_atomic[grid](
        X, Y, dot_product,
        n_elements,
        BLOCK_SIZE=block_size,
    )
    
    # Return negative dot product as distance
    return -dot_product.item()


""" Reduction-based implementation in Triton"""
@triton.jit
def dot_kernel_reduction(
    x_ptr, y_ptr, dot_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID (block ID)
    pid = tl.program_id(0)
    
    # Global thread indices
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Each thread loads one element
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Each thread computes its contribution
    xy_product = x * y
    
    # Block-level reduction (sum all threads in this block)
    dot_block = tl.sum(xy_product)
    
    # Store each block's partial sum to a separate location
    tl.store(dot_ptr + pid, dot_block)


@triton.jit
def reduce_level1_dot(
    input_ptr, output_ptr, n_blocks, n_blocks_per_group,
    BLOCK_SIZE: tl.constexpr,
):
    """
    First level reduction: Combine multiple blocks into fewer blocks
    Each block processes n_blocks_per_group values
    """
    # Program ID (block ID)
    pid = tl.program_id(0)
    
    # Calculate starting offset for this block's work
    start_idx = pid * n_blocks_per_group
    
    # Create offsets for loading data
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_blocks
    
    # Load input values
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Sum the values
    block_sum = tl.sum(values)
    
    # Store the result
    tl.store(output_ptr + pid, block_sum)


@triton.jit
def final_reduction_dot(
    dot_ptr, output_ptr, n_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    """Final reduction and dot product computation"""
    # Load all block results and reduce
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_blocks
    
    # Load values
    dots = tl.load(dot_ptr + offsets, mask=mask, other=0.0)
    
    # Sum values
    dot_sum = tl.sum(dots)
    
    # Store result (negative for distance)
    tl.store(output_ptr, -dot_sum)

def distance_dot_reduction(X, Y):
    """
    Compute dot product distance between two potentially very large vectors.
    Uses multi-level reduction to handle vectors of any size efficiently.
    
    Parameters:
    X, Y: torch.Tensor or array-like
        Input vectors to compute dot product distance between
        
    Returns:
    float: Negative dot product between X and Y (used as a distance)
    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, device='cuda', dtype=torch.float32)
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y, device='cuda', dtype=torch.float32)
    
    n_elements = X.numel()
    
    # Determine block size (power of 2)
    block_size = triton.next_power_of_2(min(n_elements, 1024))
    
    # Compute stage 1 grid size
    n_blocks_stage1 = triton.cdiv(n_elements, block_size)
    
    # Create buffer for stage 1 output
    dot_products_stage1 = torch.zeros(n_blocks_stage1, device='cuda', dtype=torch.float32)
    
    # Launch stage 1: Each thread block processes a segment of the input vectors
    dot_kernel_reduction[(n_blocks_stage1,)](
        X, Y, 
        dot_products_stage1,
        n_elements,
        BLOCK_SIZE=block_size,
    )
    
    # Check if we need multi-level reduction
    if n_blocks_stage1 <= 1024:
        # Single-level reduction is sufficient
        output = torch.zeros(1, device='cuda', dtype=torch.float32)
        final_reduction_dot[(1,)](
            dot_products_stage1,
            output, n_blocks_stage1,
            BLOCK_SIZE=1024,
        )
    else:
        # Need multi-level reduction
        # Determine blocks per group for level 1 reduction
        n_blocks_per_group = triton.next_power_of_2(min(n_blocks_stage1, 1024))
        n_blocks_stage2 = triton.cdiv(n_blocks_stage1, n_blocks_per_group)
        
        # Create buffer for stage 2
        dot_products_stage2 = torch.zeros(n_blocks_stage2, device='cuda', dtype=torch.float32)
        
        # Reduce dot products
        reduce_level1_dot[(n_blocks_stage2,)](
            dot_products_stage1, dot_products_stage2,
            n_blocks_stage1, n_blocks_per_group,
            BLOCK_SIZE=n_blocks_per_group,
        )
        
        # Check if we need another level of reduction
        if n_blocks_stage2 <= 1024:
            # Two-level reduction is sufficient
            output = torch.zeros(1, device='cuda', dtype=torch.float32)
            final_reduction_dot[(1,)](
                dot_products_stage2,
                output, n_blocks_stage2,
                BLOCK_SIZE=1024,
            )
        else:
            # Need three-level reduction
            n_blocks_per_group2 = triton.next_power_of_2(min(n_blocks_stage2, 1024))
            n_blocks_stage3 = triton.cdiv(n_blocks_stage2, n_blocks_per_group2)
            
            # Create buffer for stage 3
            dot_products_stage3 = torch.zeros(n_blocks_stage3, device='cuda', dtype=torch.float32)
            
            # Reduce dot products
            reduce_level1_dot[(n_blocks_stage3,)](
                dot_products_stage2, dot_products_stage3,
                n_blocks_stage2, n_blocks_per_group2,
                BLOCK_SIZE=n_blocks_per_group2,
            )
            
            # Final reduction
            output = torch.zeros(1, device='cuda', dtype=torch.float32)
            final_reduction_dot[(1,)](
                dot_products_stage3,
                output, n_blocks_stage3,
                BLOCK_SIZE=1024,
            )
    
    return output.item()


""" PyTorch baseline implementation """
def distance_dot_pytorch(X, Y):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, device='cuda', dtype=torch.float32)
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y, device='cuda', dtype=torch.float32)
    
    # Return negative dot product as distance
    return -torch.sum(X * Y).item()


def run_benchmark():
    # Run benchmarks for different vector sizes
    sizes = [1000, 10000, 100000, 1000000, 10000000, 100000000]
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
    print(f"CUDA Cores: {torch.cuda.get_device_properties(0).multi_processor_count}")
    print("-" * 160)
    print(f"{'Size':<10} {'CPU (ms)':<15} {'PyTorch (ms)':<15} {'CuPy (ms)':<15} {'Tri-Atomic (ms)':<15} {'Tri-Reduce (ms)':<15}  {'CPU Error':<15} {'CuPy Error':<15} {'Atomic Error':<15} {'Reduction Error':<15}")
    print("-" * 160)
    
    for size in sizes:
        # Generate random vectors
        X = torch.randn(size, device='cuda', dtype=torch.float32)
        Y = torch.randn(size, device='cuda', dtype=torch.float32)
        X_cpu = X.cpu().numpy()
        Y_cpu = Y.cpu().numpy()
        
        # Warm-up
        distance_dot_atomic(X, Y)
        distance_dot_reduction(X, Y)
        distance_dot_pytorch(X, Y)
        distance_dot_cpu(X_cpu, Y_cpu)
        distance_dot_cupy(X, Y)
        torch.cuda.synchronize()
        
        # Reference result from pytorch
        reference = distance_dot_pytorch(X, Y)
        
        # Time CPU implementation
        start = time.time()
        for _ in range(10):
            result_cpu = distance_dot_cpu(X_cpu, Y_cpu)
        end = time.time()
        time_cpu = (end - start) * 100  # ms per run
        error_cpu = abs(result_cpu - reference)
        
        # Time CuPy implementation
        start = time.time()
        for _ in range(10):
            result_cupy = distance_dot_cupy(X, Y)
        end = time.time()
        time_cupy = (end - start) * 100  # ms per run
        error_cupy = abs(result_cupy - reference)
        
        # Time atomic implementation
        start = time.time()
        for _ in range(10):
            result_atomic = distance_dot_atomic(X, Y)
            torch.cuda.synchronize()
        end = time.time()
        time_atomic = (end - start) * 100  # ms per run
        error_atomic = abs(result_atomic - reference)
        
        # Time reduction implementation
        start = time.time()
        for _ in range(10):
            result_reduction = distance_dot_reduction(X, Y)
            torch.cuda.synchronize()
        end = time.time()
        time_reduction = (end - start) * 100  # ms per run
        error_reduction = abs(result_reduction - reference)
        
        # Time PyTorch baseline
        start = time.time()
        for _ in range(10):
            result_pytorch = distance_dot_pytorch(X, Y)
            torch.cuda.synchronize()
        end = time.time()
        time_pytorch = (end - start) * 100  # ms per run
        
        print(f"{size:<10} {time_cpu:<15.4f} {time_pytorch:<15.4f} {time_cupy:<15.4f} {time_atomic:<15.4f} {time_reduction:<15.4f} {error_cpu:<15.8f} {error_cupy:<15.8f} {error_atomic:<15.8f} {error_reduction:<15.8f}")

if __name__ == "__main__":
    run_benchmark()