# DeepSeek 0.9B Mixture-of-Experts (MoE) Analysis

This directory contains the codebase for configuring, compiling, and executing a sparse 0.9B Parameter Mixture-of-Experts (MoE) Transformer layer model optimized via JAX and Flax.

## Architecture Overview
- Model Type: Sparse Mixture-of-Experts (MoE)
- Total Parameters (Memory Footprint): ~890 Million
- Active Parameters (Compute Footprint): ~265 Million (~30% active per step)
- Layer Depth: 16 Layers
- Experts Configuration: 16 Total Experts (2 Shared, 14 Routed via Top-2 Gating)
- Attention Type: Grouped-Query Attention (GQA) with 8 KV Heads

## Operational Guide
By employing conditional execution, this model decouples memory capacity from active computational overhead. The token routing mechanism relies on a linear gating layer combined with a Softmax function to map tokens to specialized expert pathways.

### Loss Optimization Note
This model uses a dual-objective loss template to ensure training stability and prevent expert collapse:
1. **Task Loss:** Standard Cross-Entropy.
2. **Auxiliary Loss ($L_{balance}$):** Implements a workload balancing penalty across all 14 routed experts.

- **Recommended Environment:** Colab T4 GPU or TPU v2 (via `jax.pmap` sharding).
- **Execution Script:** `python train.py`
