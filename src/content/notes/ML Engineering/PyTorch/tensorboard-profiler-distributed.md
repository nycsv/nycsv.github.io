---
title: TensorBoard PyTorch Profiler — Distributed Debugging
description: Using the PyTorch Profiler TensorBoard plugin to diagnose bottlenecks in distributed training
tags: [pytorch, distributed, profiling, tensorboard, debugging]
date: 2026-04-23
---

## Setup

```bash
pip install torch-tb-profiler
```

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler("./log/profiler"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(loader):
        train_step(batch)
        prof.step()
```

Each rank writes to a separate subdirectory automatically named by hostname and PID:

```
log/profiler/
├── [host]-[pid]-rank0.pt.trace.json
├── [host]-[pid]-rank1.pt.trace.json
└── ...
```

Launch TensorBoard:

```bash
tensorboard --logdir=./log/profiler
```

---

## Plugin Tabs Overview

```
TensorBoard → PyTorch Profiler plugin
├── Overview       — step time breakdown, GPU utilization summary
├── Operator       — per-op CPU/CUDA time, call count
├── Kernel         — per-CUDA-kernel time, occupancy, mean blocks/SM
├── Trace          — raw Perfetto-style timeline (opens in-browser)
├── Memory         — allocation timeline and peak usage
└── Distributed    — multi-rank comparison (the key tab for DDP)
```

---

## Overview Tab

Shows a breakdown of one training step:

```
Step time breakdown:
  DataLoader      ████░░░░░░  15%
  Forward         ██████░░░░  30%
  Backward        ████████░░  40%
  Optimizer       ██░░░░░░░░  10%
  Other           █░░░░░░░░░   5%
```

**GPU Utilization** and **GPU Tensor Core Utilization** are shown per step. Low utilization at any phase points to where to dig.

- DataLoader dominant → I/O bound
- Forward ≈ Backward but GPU util low → memory-bandwidth bound or launch overhead
- "Other" large → framework overhead, Python GIL

---

## Distributed Tab — Multi-Rank Analysis

This is the primary view for distributed debugging. It aggregates all rank traces and highlights imbalance.

### Computation/Communication Overview

```
Rank | Compute  | Comm (AllReduce) | Overlap | Comm overhead
-----|----------|------------------|---------|---------------
  0  | 520 ms   | 180 ms           | 60%     | 72 ms
  1  | 518 ms   | 182 ms           | 58%     | 76 ms
  2  | 521 ms   | 179 ms           | 61%     | 70 ms
  3  | 580 ms   | 181 ms           | 20%     | 145 ms ← straggler
```

**Overlap ratio** = how much of AllReduce runs concurrently with backward.
- High overlap → DDP bucketing is working well
- Low overlap on one rank → that rank's backward is slower (straggler) or bucket sizes are mismatched

### Exposed Communication Time

`Comm overhead = AllReduce duration × (1 - overlap ratio)`

This is the actual wall-clock time lost to communication. Minimize this, not raw AllReduce time.

### Step Time Variance

The plugin plots step time per rank across all recorded steps. A rank with consistently higher step time is a straggler — all other ranks block waiting at AllReduce.

---

## Operator Tab

Shows aggregated self-time and total-time per op across steps:

| Name | Calls | CPU Self | CUDA Self |
|---|---|---|---|
| aten::mm | 384 | 1.2 ms | 88 ms |
| aten::add_ | 1024 | 3.1 ms | 2.1 ms |
| ncclAllReduce | 96 | 0.1 ms | 210 ms |
| aten::copy_ | 256 | 8.4 ms | 12 ms |

- `aten::copy_` high CPU self → unpinned memory copies; use `pin_memory=True`
- `ncclAllReduce` CUDA self dominates → bandwidth-limited, consider gradient compression or fewer buckets
- Many `aten::add_` calls with low CUDA time → fragmented ops, candidate for `torch.compile()`

**Group by**: switch between "Operator", "Operator + Input Shape", "Source Location" to drill down.

---

## Kernel Tab

CUDA kernel-level view. Key columns:

| Column | What to check |
|---|---|
| Mean Duration | Long kernels blocking the stream |
| Occupancy | Low (<50%) → register pressure or small block size |
| Mean Blocks/SM | Low → kernel not utilizing the GPU |
| Tensor Core % | Low for matmul → not using FP16/BF16 |

A kernel with high duration but low occupancy is a prime target for precision reduction (`autocast`) or rewriting with a fused implementation.

---

## Memory Tab

Timeline of allocations and frees. Key metrics:

- **Peak Reserved** — total memory PyTorch has claimed from CUDA
- **Peak Allocated** — memory actually in use at the peak
- `Reserved - Allocated` gap = fragmentation

If OOM occurs mid-training but peak allocated looks fine, fragmentation is the cause. Fix with:

```python
torch.cuda.memory.set_per_process_memory_fraction(0.9)  # leave headroom
# or
torch.cuda.empty_cache()  # after each validation loop
```

Spikes in the allocation timeline that align with the backward pass are normal (gradient buffers). Spikes outside the backward pass suggest unexpected tensor retention (closure capturing tensors, accidental `.detach()` missing).

---

## Trace Tab (in-browser Perfetto)

Same as ui.perfetto.dev but embedded. Use this for:
- Inspecting a specific step in detail
- Verifying AllReduce overlap visually
- Finding the exact CPU op that precedes a GPU idle gap

Filter by rank using the process selector on the left panel.

---

## Workflow

```
1. Overview tab
   └── Which phase dominates step time?
       ├── DataLoader → pin_memory, more workers
       ├── Forward/Backward GPU util low → Kernel tab
       └── Large "Other" → Operator tab, look for Python overhead

2. Distributed tab
   └── Is one rank slower? → Straggler
       ├── Step time variance plot → identify which rank
       ├── Overlap ratio → low means backward is bottleneck on that rank
       └── Check that rank's GPU in Kernel tab

3. Operator tab
   └── What ops consume the most CUDA self time?
       ├── ncclAllReduce dominant → reduce bucket_cap_mb, gradient compression
       ├── aten::copy_ CPU heavy → pin_memory
       └── Many small ops → torch.compile()

4. Kernel tab
   └── Low occupancy or low Tensor Core % on matmul → switch to BF16/FP16

5. Memory tab
   └── Fragmentation? Unexpected spikes? → empty_cache, check tensor lifetimes
```

---

## Common Findings and Fixes

| Tab | Observation | Fix |
|---|---|---|
| Distributed | One rank has low overlap ratio | `bucket_cap_mb` smaller, check straggler GPU |
| Distributed | All ranks have zero overlap | Remove `find_unused_parameters=True` |
| Operator | `aten::copy_` CPU self high | `pin_memory=True`, `non_blocking=True` |
| Operator | `ncclAllReduce` >> compute time | Gradient compression, fewer all-reduces (gradient accumulation) |
| Kernel | matmul Tensor Core % = 0 | Use `torch.autocast("cuda", dtype=torch.bfloat16)` |
| Kernel | Low occupancy on attention kernels | Switch to `F.scaled_dot_product_attention` (FlashAttention backend) |
| Memory | Large Reserved - Allocated gap | `torch.cuda.memory.set_per_process_memory_fraction`, avoid fragmentation |
| Overview | DataLoader > 20% of step time | More workers, prefetch, move preprocessing offline |

---

## Multi-Node: Collecting Traces Across Machines

Each node writes traces locally. Gather them to one machine before launching TensorBoard:

```bash
# On each worker node (rank N)
rsync -av ./log/profiler/ master:/shared/log/profiler/

# On master
tensorboard --logdir=/shared/log/profiler
```

Or use a shared filesystem (NFS, GPFS, S3-mounted) as the `tensorboard_trace_handler` path directly — all ranks write to the same directory and TensorBoard reads all files at once.
