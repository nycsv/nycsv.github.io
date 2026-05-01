---
title: Reading the PyTorch Profiler in TensorBoard
description: How to read each profiler view, identify the four major training bottlenecks, and take concrete steps to fix them
tags: [pytorch, profiling, tensorboard, performance, cuda, distributed]
date: 2026-04-23
---

The TensorBoard PyTorch Profiler plugin (`torch-tb-profiler`) gives you five views of a training run. Each one is best for a different class of problem.

---

## Setup

```bash
pip install torch-tb-profiler
```

```python
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3),
    on_trace_ready=tensorboard_trace_handler("./tb_logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()
```

```bash
tensorboard --logdir ./tb_logs
```

---

## The Five Views

```
TensorBoard → PyTorch Profiler plugin
├── Overview       — step time breakdown, GPU utilization summary
├── Operator       — per-op CPU/CUDA time, call count
├── Kernel         — per-CUDA-kernel time, occupancy, Tensor Core usage
├── Trace          — raw Perfetto-style timeline
├── Memory         — allocation timeline and peak usage
└── Distributed    — multi-rank comparison (the key tab for DDP)
```

---

## Overview Tab

Your starting point. The step time breakdown shows:

```
DataLoader      ████░░░░░░  15%
Forward         ██████░░░░  30%
Backward        ████████░░  40%
Optimizer       ██░░░░░░░░  10%
Other           █░░░░░░░░░   5%
```

**GPU Utilization** is the headline metric. Below ~80% means the GPU is waiting on something.

- DataLoader dominant → I/O bound, fix the pipeline
- Forward ≈ Backward, low GPU util → memory-bandwidth bound or launch overhead
- "Other" large → Python GIL or framework overhead

---

## Operator Tab

Per-op breakdown. Sort by **Self Time** (the op itself, not its children) to find the actual bottleneck rather than the wrapper.

Key things to look for:

| Op | High what? | Meaning |
|---|---|---|
| `aten::copy_` | CPU self time | Unpinned memory → use `pin_memory=True` |
| `ncclAllReduce` | CUDA self time | Bandwidth-limited communication |
| many `aten::add_` | low CUDA, many calls | Fragmented ops → `torch.compile()` |

Switch "Group by" to **Source Location** to see which lines of your code are most expensive.

---

## Kernel Tab

CUDA kernels on the hardware, not PyTorch function names. Sort by **Total Duration**.

| Column | What to look for |
|---|---|
| Occupancy | Low (<50%) → register pressure or small block size |
| Tensor Core % | 0% on matmul → you're in FP32, not using hardware properly |
| Mean Blocks/SM | Low → kernel not saturating the GPU |

A matmul kernel with `Tensor Core % = 0` means you're leaving most of the GPU's compute on the table. Fix: enable AMP with `torch.autocast("cuda", dtype=torch.bfloat16)`.

---

## Memory Tab

Timeline of allocations and frees.

- **Peak Reserved** — total memory PyTorch has claimed from CUDA
- **Peak Allocated** — memory actually in use at peak

A large `Reserved - Allocated` gap is fragmentation. The allocator is holding blocks it isn't using. Fix:

```python
torch.cuda.empty_cache()  # after validation, between phases
```

Allocation spikes during backward are normal (gradient buffers). Spikes *outside* backward suggest unexpected tensor retention — check for closures or missing `.detach()`.

---

## Distributed Tab

The primary view for multi-rank debugging. It aggregates all rank traces and highlights imbalance.

```
Rank | Compute  | Comm (AllReduce) | Overlap | Comm overhead
-----|----------|------------------|---------|---------------
  0  | 520 ms   | 180 ms           | 60%     | 72 ms
  1  | 518 ms   | 182 ms           | 58%     | 76 ms
  3  | 580 ms   | 181 ms           | 20%     | 145 ms  ← straggler
```

**Overlap ratio** = how much of AllReduce runs concurrently with backward. Low on one rank means that rank's backward is slower or its bucket sizes are off.

**Exposed comm time** = `AllReduce duration × (1 - overlap ratio)` — the actual wall-clock time lost. This is what you want to minimize.

---

## Bottlenecks and Fixes

### DataLoader (I/O bound)

```python
DataLoader(dataset,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)
```

### CPU Overhead

```python
# Bad — forces CPU/GPU sync every step
running_loss += loss.item()

# Good — accumulate tensor, sync periodically
running_loss += loss
if step % 100 == 0:
    print(running_loss.item())

# Fuse ops to reduce kernel launches
model = torch.compile(model)
```

### GPU Compute (FP32, Tensor Cores unused)

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()
with autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = model(batch)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Communication (AllReduce bottleneck)

```python
# Gradient accumulation — fewer all-reduces
for i, batch in enumerate(loader):
    loss = model(batch) / accum_steps
    loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Tune bucket size for your network bandwidth
model = DDP(model, bucket_cap_mb=25)  # default 25 MB, try smaller for fast networks
```

---

## Quick Reference

| Symptom | View | Likely cause | First fix |
|---|---|---|---|
| GPU util < 80% | Overview → Trace | DataLoader or CPU overhead | `num_workers`, `pin_memory` |
| Fragmented GPU kernels | Trace | CPU dispatch overhead | `torch.compile`, remove `.item()` |
| Tensor Cores = 0% | Kernel | FP32 compute | Enable AMP |
| High NCCL time | Distributed | Gradient sync bottleneck | Gradient accumulation, tune `bucket_cap_mb` |
| Large Reserved-Allocated gap | Memory | Fragmentation | `empty_cache()` between phases |
