# 590A Assignment3

## Overview 
This assignment has the following objectives:
- Learn about the basic components of the TPU architecture essential for writing efficient TPU programs. 
- Getting familiar with writing and evaluating TPU programs in Pallas.
- Navigating memory management and performance trade-offs when writing fused tensor operators in Pallas.



## Section 1: TPU architecture fundamentals (10 points)
*This section is write-up only. No experiments to run.*

TPUs are specialized hardware accelerators developed by Google to accelerate machine learning workloads. While GPUs are accelerators designed for a wide range of parallel operations, TPUs are specifically optimized for massive matrix operations, making them particularly efficient for training / inference workloads. As such, TPUs and GPUs differ significantly in their architecture and how tensor operations are scheduled on the hardware. Understanding the underlying hardware architecture is important to write performant kernels (for both GPUs and TPUs). This assignment focuses on learning about the TPU architecture fundamentals and writing efficient kernels using a high-level python library called Pallas.

The following documentation might be helpful in answering the questions:
- [Writing TPU kernels with Pallas](https://docs.jax.dev/en/latest/pallas/tpu/details.html#what-is-a-tpu)
- [Software Pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html)
- [TPU Pipelining](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html)
- [Matrix Multiplication](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html)

Q1: Briefly describe the following components of the memory-heirarchy in TPUs: HBM, VMEM, SMEM, VREG, SREG. What is the size of HBM, VMEM, and SMEM in TPU v5e? 

Q2: Briefly describe the various compute units in a TPU. What is the memory bandwidth between HBM and the compute units in TPU v5e? What is the memory bandwidth (exact value not required) between VMEM and the compute units for TPU v5e?  

Q3: What is software-pipelining? What purpose does it serve in writing efficient TPU (or GPU) kernels? What abstractions does Pallas expose to manage pipelining in TPUs? Very briefly describe each abstraction. What is the trade-off involved in pipeline performance when choosing small versus large block sizes? 

## Section 2: Implementing the SiLU operator in Pallas (40 points)
In this section, you will implement and evaluate a SiLU operator in Pallas. SiLU (aka swish) is an activation function used in MLP to introduce non-linearity, improving the neural network's capability of modeling complex relationships in training data. SiLU is mathematically defined as:

$$\text{SiLU}(x) = \frac{x}{1 + e^{-x}}$$

Reference: [JAX implementation of SiLU](https://docs.jax.dev/en/latest/_autosummary/jax.nn.silu.html)

Q1: Given an input matrix of size `(8192, 8192)`, theoretically infer the number of memory accesses (between HBM and compute units) required to compute SiLU. What is the total size of the memory accesses in bytes? Assume bf16 datatype. 

Q2: Implement the following in the provided code template: 
- `silu_naive` using basic math operations. Do not use inbuilt JAX library function for SiLU. We use that as a reference for our implementation.
- `silu_sigmoid` using inbuilt `jax.sigmoid`. 
- `silu_pallas` kernel using pallas. 

Profile each implementation using `xprof` and answer the following questions:

(A) Report the number of memory allocations / deallocations observed in each implementation. 

(B) Report the name of the jit operations observed in each implementation.

(C) Report the observed memory bandwidth for each implementation. Use the value printed by `silu.py` for this question. Is the implementation memory-bound or compute bound? Why? 

(D) Rank the SiLU implementations from lowest memory bandwidth to highest. Explain the ordering observed. The observed profiling data above should provide a hint about the observed performance. 

## Section 3: Memory Efficient Fused-MLP implementation in Pallas (50 points)
*Note: You are required to generate your own plotting scripts for this section.*

In this section, you will implement a memory efficient fused-MLP layer using Pallas. 
In this assignment, a fused MLP layer is described as the sequence of Linear --> SiLU --> Linear operations. 
Let `H` be the size of the hidden dimension. Let `W1` and `W2` be fixed weight matrices, `X` be the input matrix, and `Y` be the output matrix. Then the MLP operation is mathematically defined as: 

`Y = SiLU(X @ W1) @ W2`

where each matrix has the following shapes:

`X: (M, K), W1: (K, H), W2: (H, N), Y: (M, N)`


Q1: Derive the number of floating point operations in terms of M, K, H, and N for the fused MLP operation above. You will use this value to compute FLOPs utilzation for the following questions.

Implement a naive version of the fused MLP operation by completing `make_fused_pallas_naive` in the provided template code. The naive version defines a 1D grid over the input dimension M. The remaining dimensions (K, H, and N) are fully materialized for each tile being computed on the TPU. 

Q2: Keeping the dimensions `(M, K, N) = (4096, 1024, 1024)` fixed, execute the kernel for `BM=1024` and `H = [512, 1024, 2048, 4096, 8192, 16384]`. Beyond which value of H does the kernel exhaust resources? Why? For the largest successful value of H, sweep across various block sizes and compute the FLOPs utilization (number of floating point ops / execution time) for each. How does the FLOPs utilization vary with block sizes? Why? 

Implement a memory optimized version of fused MLP operation by defining a 2D grid over the M and H dimensions. This means that `W1` and `W2` are only partially materialized for each *tile* being computed on the TPU. More details about tiling are as follows. 

### Tiling

Given block sizes `BM` and `BH`, each kernel instance operates on the following tiles:

| Tile | Shape | Description |
|------|-------|-------------|
| `x`  | `[BM, K]` | A row-block of the input `X` |
| `w1` | `[K, BH]` | A column-slice of `W1` |
| `w2` | `[BH, N]` | A row-slice of `W2` |
| `y`  | `[BM, N]` | A row-block of the output `Y` |

Number of `M` blocks: `num_m_blocks = M / BM`

Number of `H` blocks: `num_h_blocks = H / BH`

Each kernel instance computes one tile of size `[BM, BH]` and its partial contribution to a `[BM, N]` tile of `Y`. There are `num_m_blocks * num_h_blocks` such tiles. 

The full output tile is recovered by accumulating contributions across all `BH`-slices of `H`. For all tiles with first index `m_i` (which would be a value between 0 and `num_m_blocks-1`), this can be accomplished by initializing an accumulator in the `VMEM` and accumulating output from tiles `(m_i, 0), (m_i, 1), (m_i, 2), ..., (m_i, num_h_blocks-1)`. The resulting dimension of this accumulator for a given `m_i` and across all tiles of the `H` dimension is `[BM, N]`. Then, results from all such `num_m_blocks` tiles with dimension `[BM, N]` are naturally accumulated by defining the `out_specs` accordingly in the `pallas_call` for this kernel.

*Hint: You can take help of pl.program_id() inside the kernel body to know which tile is being computed. You want to use a "scratch" accumulator inside the kernel body to accumulate the computation across the `H` dimension for a given `m_i`.*

Verify the correctness of the implementation by evaluating the kernel against the naive implementation for different values of (M, K, H, and N). Refer to Section 2 template code for an example of correctness checking. 

Q3: For the following dimensions `M, K, H, N = (4096, 1024, 16384, 1024)`, run a 2D sweep using the following block sizes for M and H: 
`Block sizes for M = [8, 16, 32, 64, 128, 256, 512, 1024]`
`Block sizes for H = [128, 256, 512, 1024, 2048]`
Plot a 2D heatmap showing block sizes for M on the y-axis and block sizes for H on the x-axis and computing TFLOPs utilization for each. For tile sizes where the kernel exhausts resources, mark it with `OOM`. Which tile size achieves the highest FLOPs utilization? Why? 








