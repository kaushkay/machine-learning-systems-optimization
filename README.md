# Machine Learning Systems Optimization

This repository contains optimizations for two critical components of modern machine learning systems: GPU-accelerated vector operations and distributed inference pipelines.

## Overview

### Task 1: GPU-Accelerated Distance Functions
High-performance implementations of distance metrics (L2, L1, Cosine, Dot Product) using Triton kernels, achieving significant speedups over CPU baselines through hierarchical reduction techniques and memory optimizations.

### Task 2: RAG Service Optimization
Enhanced Retrieval-Augmented Generation service with request queuing, dynamic batching, and load balancing mechanisms, improving throughput and reducing latency under high concurrency.

## Project Structure

```
├── task1_gpu_distance_functions/    # GPU-accelerated distance computations
├── task2_rag_serving/              # RAG service optimizations
└── docs/                           # Documentation and analysis
```

## Quick Start

### Prerequisites
- NVIDIA GPU with CUDA support
- Python 3.8+
- PyTorch
- Triton
- FastAPI
- Required dependencies (see requirements.txt)

### Installation
```bash
git clone https://github.com/yourusername/machine-learning-systems-optimization.git
cd machine-learning-systems-optimization
pip install -r requirements.txt
```

## Task 1: GPU Distance Functions

Navigate to `task1_gpu_distance_functions/` for GPU-accelerated distance computations:

```bash
cd task1_gpu_distance_functions
python l2_squared.py    # Run L2 distance benchmarks
python cosine.py        # Run cosine distance benchmarks
python dot_product.py   # Run dot product benchmarks
python l1.py           # Run L1 distance benchmarks
```

**Key Features:**
- Up to 142× speedup over CPU baselines for large vectors (100M elements)
- Hierarchical reduction techniques for optimal GPU utilization
- Support for vectors exceeding GPU memory through chunked processing
- Atomic and hierarchical reduction implementations

## Task 2: RAG Service

Navigate to `task2_rag_serving/` for optimized RAG implementations:

### Running Services

**Baseline Implementation:**
```bash
python baseline/serving_rag_baseline.py
```

**Request Queue Optimization:**
```bash
python optimized/serving_rag_request_queue.py
```

**Batch Processing Optimization:**
```bash
python optimized/serving_rag_batch.py
```

**Distributed Load Balancer:**
```bash
# Start load balancer
python distributed/load_balancer.py

# Start multiple RAG servers
python distributed/rag_server.py --port 8001
python distributed/rag_server.py --port 8002
python distributed/rag_server.py --port 8003
```

### Performance Testing

**Test Basic Implementation:**
```bash
python testing/test_script.py --url http://localhost:8000 --requests 50 --concurrency "1,5,10,20"
```

**Test Optimized Implementations:**
```bash
python testing/test_script.py --url http://localhost:8000 --requests 50 --concurrency "1,5,10,20" --optimized
```

**Test Load Balancer:**
```bash
python testing/test_script_LB.py --url http://localhost:8000 --requests 100 --concurrency "1,5,10,20"
```

## Performance Results

### GPU Distance Functions
- **L2 Distance**: Up to 142× speedup over CPU for 100M-element vectors
- **Cosine Distance**: Optimal performance with hierarchical reduction for large vectors
- **Memory Efficiency**: 80% theoretical memory bandwidth utilization

### RAG Service Optimizations
- **Throughput**: Up to 38.8% improvement with batch processing
- **Latency**: 45.1% reduction under high concurrency
- **Scalability**: Maintains 100% success rate across all concurrency levels

## Key Innovations

1. **Hierarchical Reduction Kernels**: Multi-stage reduction for handling arbitrarily large vectors
2. **Dynamic Batch Formation**: Hybrid time/size-based batching policy
3. **Load Balancing**: Round-robin distribution with health monitoring
4. **Memory Optimization**: Chunked processing and efficient GPU memory management

## Hardware Requirements

- **GPU**: NVIDIA RTX A6000 or equivalent (tested configuration)
- **Memory**: Sufficient GPU memory for your dataset size
- **CUDA**: Compute Capability 8.6 or higher recommended

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This work was developed as part of the Machine Learning Systems course (INFR11269) focusing on optimizing both computational efficiency and operational scalability in modern ML deployments.
