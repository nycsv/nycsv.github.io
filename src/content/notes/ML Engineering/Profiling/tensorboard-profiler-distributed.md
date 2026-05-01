---
title: TensorBoard PyTorch Profiler — Distributed Debugging
description: Using the PyTorch Profiler TensorBoard plugin to diagnose bottlenecks in DDP/multi-node training, with a focus on the Distributed tab
tags: [pytorch, distributed, profiling, tensorboard, ddp, debugging]
date: 2026-04-23
---

The TensorBoard profiler plugin's **Distributed tab** aggregates traces from all ranks in one view, making it easy to spot straggler ranks and communication overhead without manually comparing JSON files.

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

Each rank writes to a separate file named by hostname and PID:

```
log/profiler/
├── [host]-[pid]-rank0.pt.trace.json
├── [host]-[pid]-rank1.pt.trace.json
└── ...
```

```bash
tensorboard --logdir=./log/profiler
```

---

## The Distributed Tab

This is the main view for multi-rank debugging. It shows per-rank compute/communication breakdown and highlights imbalance.

### Computation / Communication Breakdown

```
Rank | Compute  | Comm (AllReduce) | Overlap | Comm overhead
-----|----------|------------------|---------|---------------
  0  | 520 ms   | 180 ms           | 60%     | 72 ms
  1  | 518 ms   | 182 ms           | 58%     | 76 ms
  2  | 521 ms   | 179 ms           | 61%     | 70 ms
  3  | 580 ms   | 181 ms           | 20%     | 145 ms  ← straggler
```

**Overlap ratio** = fraction of AllReduce that runs concurrently with backward. High is good — it means DDP bucketing is working.

**Exposed comm time** = `AllReduce duration × (1 - overlap ratio)`. This is what you actually lose to communication.

Low overlap on one rank means that rank's backward is slower (straggler) or bucket sizes aren't aligned to layer boundaries.

### Step Time Variance

The plugin plots step time per rank across all recorded steps. A rank with consistently higher step time will make every other rank wait at the AllReduce barrier.

---

## Overview Tab

Starting point for any profiling session. The step time breakdown shows which phase dominates:

```
DataLoader      ████░░░░░░  15%
Forward         ██████░░░░  30%
Backward        ████████░░  40%
Optimizer       ██░░░░░░░░  10%
Other           █░░░░░░░░░   5%
```

- **DataLoader dominant** → I/O bound
- **Low GPU utilization with normal forward/backward split** → launch overhead or memory-bandwidth bound
- **"Other" large** → Python/framework overhead, GIL contention

---

## Operator Tab

Aggregated self-time per op across steps. Sort by **CUDA Self Time** to find what the GPU is actually spending time on.

| Op | Pattern | Action |
|---|---|---|
| `aten::copy_` high CPU self | Unpinned memory copies | `pin_memory=True` |
| `ncclAllReduce` dominates CUDA | Bandwidth-limited | Gradient compression, fewer buckets |
| Many `aten::add_` with low CUDA | Fragmented ops | `torch.compile()` |

Switch "Group by" to **Source Location** to trace expensive ops back to your code.

---

## Kernel Tab

CUDA kernel-level view. Key columns:

| Column | Check for |
|---|---|
| Mean Duration | Long kernels blocking the stream |
| Occupancy | < 50% → register pressure or small block size |
| Tensor Core % | 0% on matmul → not using FP16/BF16 |

A matmul kernel with `Tensor Core % = 0` means you're running FP32. Switching to `autocast(dtype=torch.bfloat16)` can give 2–4× kernel throughput improvement.

---

## Memory Tab

- **Peak Reserved** — total memory PyTorch holds from CUDA (includes free pool)
- **Peak Allocated** — memory in active use at the peak

A large `Reserved - Allocated` gap is fragmentation. The allocator has blocks it isn't using. Fix:

```python
torch.cuda.empty_cache()          # after validation
# or
torch.cuda.memory.set_per_process_memory_fraction(0.9)  # leave headroom
```

---

## Multi-Node: Collecting Traces

Each node writes traces locally. Gather them before launching TensorBoard:

```bash
# On each worker node
rsync -av ./log/profiler/ master:/shared/log/profiler/

# On master
tensorboard --logdir=/shared/log/profiler
```

Or point `tensorboard_trace_handler` at a shared filesystem (NFS, GPFS) directly — all ranks write to the same directory and TensorBoard reads all files at once.

---

## Common Findings and Fixes

| Tab | Observation | Fix |
|---|---|---|
| Distributed | One rank has low overlap ratio | Smaller `bucket_cap_mb`, check straggler GPU |
| Distributed | All ranks have zero overlap | Remove `find_unused_parameters=True` |
| Operator | `aten::copy_` CPU self high | `pin_memory=True`, `non_blocking=True` |
| Operator | `ncclAllReduce` >> compute | Gradient accumulation, fewer all-reduces |
| Kernel | matmul Tensor Core % = 0 | `torch.autocast("cuda", dtype=torch.bfloat16)` |
| Kernel | Low occupancy on attention | `F.scaled_dot_product_attention` (FlashAttention) |
| Memory | Large Reserved - Allocated gap | `empty_cache()`, avoid fragmentation |
| Overview | DataLoader > 20% of step | More workers, prefetch, offline preprocessing |
