import cupy as cp
import triton
import triton.language as tl
import numpy as np
import torch
import time
from sklearn.metrics.pairwise import manhattan_distances


def manhattan_cupy(X_cupy, Y_cupy):
    return float(cp.sum(cp.abs(X_cupy - Y_cupy)))

def manhattan_cupy_chunked(X_cupy, Y_cupy, chunk_size=10000000):
    n_elements = X_cupy.shape[0]
    total_sum = 0.0
    
    for start_idx in range(0, n_elements, chunk_size):
        end_idx = min(start_idx + chunk_size, n_elements)
        chunk_X = X_cupy[start_idx:end_idx]
        chunk_Y = Y_cupy[start_idx:end_idx]
        abs_diff = cp.abs(chunk_X - chunk_Y)
        chunk_sum = cp.sum(abs_diff)
        total_sum += float(chunk_sum)
        
    return total_sum
        

def manhattan_pytorch(X_torch, Y_torch):
    return torch.sum(torch.abs(X_torch - Y_torch))

@triton.jit
def manhattan_triton_atomic(X_ptr, Y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(Y_ptr + offsets, mask=mask, other=0.0)
    
    abs_diff = tl.abs(x - y)
    
    block_sum = tl.sum(abs_diff, axis=0)
    tl.atomic_add(output_ptr, block_sum)

@triton.jit
def manhattan_triton_hierarchical_stage_1(X_ptr, Y_ptr, block_sums_ptr, n_elements, BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(Y_ptr + offsets, mask=mask, other=0.0)
    
    abs_diff = tl.abs(x - y)
    
    block_sum = tl.sum(abs_diff, axis=0)
    tl.store(block_sums_ptr + pid, block_sum)

@triton.jit    
def manhattan_triton_hierarchical_stage_2(block_sums_ptr, output_ptr, n_blocks, BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(axis=0)
    
    if pid == 0:  # Only the first thread block performs reduction
        # Initialize sum
        final_sum = 0.0
        
        # Sequential reduction instead of atomic add
        for i in range(n_blocks):
            final_sum += tl.load(block_sums_ptr + i)
            
        # Store the final result
        tl.store(output_ptr, final_sum)


def manhattan_sklearn(X, Y):
    if not isinstance(X, np.ndarray):
        X = np.asarray(X, dtype=np.float32).reshape(1, -1)
    else:
        X = X.reshape(1, -1)
        
    if not isinstance(Y, np.ndarray):
        Y = np.asarray(Y, dtype=np.float32).reshape(1, -1)
    else:
        Y = Y.reshape(1, -1)
        
    dist = manhattan_distances(X, Y)[0, 0]
    
    return float(dist)

def distance_manhattan_atomic(X_torch, Y_torch):
    X_triton = X_torch.reshape(-1)
    Y_triton = Y_torch.reshape(-1)
    
    n_elements = X_triton.numel()
    
    output = torch.zeros(1, device="cuda", dtype=torch.float32)
    
    block_size = triton.next_power_of_2(min(n_elements, 1024))
    
    grid = (triton.cdiv(n_elements, block_size),)
    
    # warm_up_triton_atomic
    manhattan_triton_atomic[grid](X_triton, Y_triton, output, n_elements, BLOCK_SIZE=block_size)
    torch.cuda.synchronize()
    
    
    # Benchmark Triton Atomic
    times = []
    for _ in range(10):
        output.zero_()  # Outside the timed section
        torch.cuda.synchronize()  # Ensure zeroing is complete
        
        start = time.time()
        manhattan_triton_atomic[grid](X_triton, Y_triton, output, n_elements, BLOCK_SIZE=block_size)
        torch.cuda.synchronize()
        end = time.time()
        
        times.append((end - start) * 1000)  # ms

    time_triton_atomic = sum(times) / len(times)
    result_triton_atomic = output.item()
    error_triton_atomic = result_triton_atomic
    return result_triton_atomic, error_triton_atomic, time_triton_atomic

if __name__ == "__main__":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
    print(f"CUDA Cores: {torch.cuda.get_device_properties(0).multi_processor_count}")
    print("-" * 160)
    
    
    size = 1_000_000
    
    X_cupy = cp.random.randn(size, dtype=cp.float32)
    Y_cupy = cp.random.randn(size, dtype=cp.float32)
    
    X_torch = torch.tensor(X_cupy, device='cuda', dtype=torch.float32)
    Y_torch = torch.tensor(Y_cupy, device='cuda', dtype=torch.float32)
    
    """Reference Implementation in SkLearn"""
    # Convert to NumPy for reference
    X_cpu = X_torch.cpu().numpy()
    Y_cpu = Y_torch.cpu().numpy()
    
    # Reference computation with sklearn
    start = time.time()
    reference = manhattan_sklearn(X_cpu, Y_cpu)
    end = time.time()
    reference_time = (end - start) * 1000 

    """Standard Cupy Implementation"""
    # warm_up_cupy_standard
    manhattan_cupy(X_cupy, Y_cupy)
    cp.cuda.stream.get_current_stream().synchronize()
    
    # Benchmark Standard Cupy
    start = time.time()
    for _ in range(10):
        result_cupy_standard = manhattan_cupy(X_cupy, Y_cupy)
        cp.cuda.stream.get_current_stream().synchronize()
    end = time.time()
    time_cupy_standard = ((end - start) * 1000) / 10  # ms per run
    error_cupy_standard = result_cupy_standard - reference
    
    """Chunked Cupy Implementation"""
    # warm_up_cupy_chunked
    manhattan_cupy_chunked(X_cupy, Y_cupy)
    cp.cuda.stream.get_current_stream().synchronize()
    
    # Benchmark Chunked Cupy
    start = time.time()
    for _ in range(10):
        result_cupy_chunked = manhattan_cupy_chunked(X_cupy, Y_cupy, chunk_size=10000000)
        cp.cuda.stream.get_current_stream().synchronize()
    end = time.time()
    time_cupy_chunked = ((end - start) * 1000) / 10  # ms per run
    error_cupy_chunked = result_cupy_chunked - reference
    
    
    """Pytorch Implementation"""
    # warm_up_torch_standard
    manhattan_pytorch(X_torch, Y_torch)
    torch.cuda.synchronize()
    
    # Benchmark Torch Standard
    start = time.time()
    for _ in range(10):
        result_torch_standard = manhattan_pytorch(X_torch, Y_torch)
        torch.cuda.synchronize()
    end = time.time()
    time_torch_standard = ((end - start) * 1000) / 10
    error_torch_standard = result_torch_standard - reference
    
    
    """Triton Atomic Implementation"""
    
    X_triton = X_torch.reshape(-1)
    Y_triton = Y_torch.reshape(-1)

    n_elements = X_triton.numel()
    
    # output = torch.zeros(1, device="cuda", dtype=torch.float32)
    
    block_size = triton.next_power_of_2(min(n_elements, 1024))
    
    result_triton_atomic, error_triton_atomic, time_triton_atomic = distance_manhattan_atomic(X_torch, Y_torch)
    error_triton_atomic -= reference
    
    """Triton Hierarchical Implementation"""
    n_blocks = triton.cdiv(n_elements, block_size)
    block_sums = torch.zeros(n_blocks, device="cuda", dtype=torch.float32)
    output = torch.zeros(1, device="cuda", dtype=torch.float32)
    
    # warm_up_triton_hierarchical
    manhattan_triton_hierarchical_stage_1[(n_blocks,)](X_triton, Y_triton, block_sums, n_elements, BLOCK_SIZE=block_size)
    if n_blocks > block_size:
        n_reduction_blocks = triton.cdiv(n_blocks, block_size)
        manhattan_triton_hierarchical_stage_2[(n_reduction_blocks,)](
            block_sums,
            output,
            n_blocks,
            BLOCK_SIZE=block_size,
        )
        torch.cuda.synchronize()
    else:
        manhattan_triton_hierarchical_stage_2[(1,)](
            block_sums,
            output,
            n_blocks,
            BLOCK_SIZE=block_size,
        )
        torch.cuda.synchronize()
        
    # Benchmark Triton Hierarchical
    times = []
    for _ in range(10):
        output.zero_()  # Outside the timed section
        torch.cuda.synchronize()  # Ensure zeroing is complete
        
        start = time.time()
        manhattan_triton_hierarchical_stage_1[(n_blocks,)](X_triton, Y_triton, block_sums, n_elements, BLOCK_SIZE=block_size)
        if n_blocks > block_size:
            n_reduction_blocks = triton.cdiv(n_blocks, block_size)
            manhattan_triton_hierarchical_stage_2[(n_reduction_blocks,)](
                block_sums,
                output,
                n_blocks,
                BLOCK_SIZE=block_size,
            )
            torch.cuda.synchronize()
        else:
            manhattan_triton_hierarchical_stage_2[(1,)](
                block_sums,
                output,
                n_blocks,
                BLOCK_SIZE=block_size,
            )
            torch.cuda.synchronize()
        end = time.time()
        
        times.append((end - start) * 1000)  # ms

    time_triton_hierarchical = sum(times) / len(times)
    result_triton_hierarchical = output.item()
    error_triton_hierarchical = result_triton_hierarchical - reference
    
    print(f"Size {size}")
    print(f"Reference Result: {reference}. Time Performance with SKLearn: {reference_time} ms.")
    print("-" * 80)
    print(f"Cupy Standard Result: {result_cupy_standard} with the error of {error_cupy_standard}. Time Performance: {time_cupy_standard} ms.")
    print(f"Cupy Chunked Result: {result_cupy_chunked} with the error of {error_cupy_chunked}. Time Performance: {time_cupy_chunked} ms.")
    print(f"Pytorch Standard Result: {result_torch_standard} with the error of {error_torch_standard}. Time Performance: {time_torch_standard} ms.")
    print(f"Triton Atomic Result: {result_triton_atomic} with the error of {error_triton_atomic}. Time Performance: {time_triton_atomic} ms.")
    print(f"Triton Hierarchical Result: {result_triton_hierarchical} with the error of {error_triton_hierarchical}. Time Performance: {time_triton_hierarchical} ms.")

    print("-" * 80)
    
    # Memory usage info
    print("\nMemory Information:")
    print("-" * 80)
    print(f"Input vector size (each): {size * 4 / (1024**2):.2f} MB (float32)")
    print(f"Total GPU memory usage (approximate): {3 * size * 4 / (1024**2):.2f} MB")