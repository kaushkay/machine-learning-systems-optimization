# Task 1: GPU-Accelerated Distance Functions

This directory contains optimized implementations of various distance metrics using GPU acceleration with Triton kernels.

## Overview

Implements four essential distance metrics with both atomic and hierarchical reduction approaches:

- **L2 Squared Distance** (`l2_squared.py`)
- **L1 Manhattan Distance** (`l1.py`)
- **Cosine Distance** (`cosine.py`)
- **Dot Product Distance** (`dot_product.py`)

## Key Features

- **Multiple Implementation Strategies**: CPU baseline, PyTorch, CuPy, and custom Triton kernels
- **Hierarchical Reduction**: Multi-stage reduction for handling vectors of any size
- **Memory Optimization**: Chunked processing for vectors exceeding GPU memory
- **Automatic Benchmarking**: Performance comparison across all implementations
- **Numerical Stability**: Epsilon-based threshold checks and zero-filling for out-of-bounds access

## Implementation Details

### Triton Optimizations

1. **Hierarchical Reduction**:
   - Dynamically adapts to vector size
   - Up to 3-level reduction for extremely large vectors
   - Minimizes global memory access and thread synchronization

2. **Atomic Operations**:
   - Immediate aggregation for certain metrics (especially L1)
   - Eliminates intermediate result storage overhead

3. **Memory Patterns**:
   - Register-level computations
   - Empirically tuned thread block sizes
   - Coalesced memory access patterns

## Usage

### Running Benchmarks

Each file can be executed directly to run comprehensive benchmarks:

```bash
# L2 Squared Distance
python l2_squared.py

# L1 Manhattan Distance
python l1.py

# Cosine Distance
python cosine.py

# Dot Product Distance
python dot_product.py
```

### Modifying Vector Sizes

The vector dimensions are configurable in each file. Look for the `size` variable:

```python
# In l2_squared.py and l1.py
size = 1_000_000  # Modify this value

# In cosine.py and dot_product.py
sizes = [1000, 10000, 100000, 1000000, 10000000, 100000000]  # Modify this list
```

## Performance Results

### Benchmark Output Format

```
GPU Name: NVIDIA RTX A6000
Memory Total: 48.83 GB
CUDA Compute Capability: 8.6
CUDA Cores: 84
--------------------------------------------------------------------------------
Size       CPU (ms)       PyTorch (ms)   CuPy (ms)      Tri-Atomic (ms) Tri-Reduce (ms)  CPU Error      CuPy Error     Atomic Error   Reduction Error
--------------------------------------------------------------------------------
1000       3.044          0.1121         time_cupy      0.1514          0.2331           error_cpu      error_cupy     error_atomic   error_hierarchical
```

### Performance Characteristics

- **Small vectors (<100K)**: CPU/PyTorch often outperform due to kernel launch overhead
- **Medium vectors (100K-1M)**: PyTorch typically leads
- **Large vectors (>10M)**: Triton implementations show significant advantages
- **Memory bandwidth**: Triton achieves ~80% of theoretical bandwidth vs 56% for PyTorch/CuPy

## Implementation Strategy by Metric

### L2 Distance
- **Best approach**: Hierarchical reduction for all sizes
- **Speedup**: Up to 142× over CPU for 100M elements
- **Memory pattern**: Squared differences with efficient reduction

### L1 Distance
- **Best approach**: Atomic operations consistently outperform hierarchical
- **Reason**: Simple absolute value benefits from immediate aggregation
- **Memory pattern**: Minimal intermediate storage

### Cosine Distance
- **Best approach**: Hierarchical for large vectors, atomic for smaller
- **Complexity**: Requires three separate reductions (dot product, X-norm, Y-norm)
- **Crossover point**: ~10M-100M elements

### Dot Product
- **Best approach**: Both approaches perform similarly
- **Simplicity**: Single reduction operation
- **Performance**: Slight edge to hierarchical at largest scales

## Error Analysis

All implementations include numerical verification against reference implementations:

- **CPU Reference**: Scikit-learn implementations where available
- **Error Tolerance**: Typically < 1e-6 for numerical differences
- **Stability Checks**: Epsilon thresholds for division operations

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support
- **Memory**: Varies by vector size (see memory usage output)
- **CUDA**: Compute Capability 6.0+ recommended
- **Driver**: Compatible with PyTorch CUDA version

## Technical Notes

### Memory Usage
The benchmarks report approximate GPU memory usage:
```
Input vector size (each): X.XX MB (float32)
Total GPU memory usage (approximate): X.XX MB
```

### Optimization Techniques
1. **Block Size Tuning**: `triton.next_power_of_2(min(n_elements, 1024))`
2. **Grid Size Calculation**: `triton.cdiv(n_elements, block_size)`
3. **Memory Coalescing**: Contiguous memory access patterns
4. **Numerical Stability**: Branch-free conditional operations

### Chunked Processing
For vectors exceeding GPU memory:
```python
def chunked_processing(X, Y, chunk_size=10000000):
    # Process in chunks to handle memory constraints
    total_sum = 0.0
    for start_idx in range(0, n_elements, chunk_size):
        # Process chunk...
```

This design enables processing of arbitrarily large vectors that exceed GPU memory capacity.
