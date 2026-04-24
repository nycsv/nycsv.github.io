---
title: Reading the PyTorch Profiler in TensorBoard
description: How to read each profiler view, identify the four major bottlenecks, and take concrete steps to improve training performance
tags: [pytorch, profiling, tensorboard, performance, cuda, distributed]
date: 2026-04-23
---

## Setup

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

## Part 1: Reading Each View

### 1. Overview

Your starting point. The **Step Time Breakdown** pie chart divides wall time into:
- CPU compute
- GPU compute
- DataLoader
- CPU → GPU transfer
- Other

**Key metric: GPU Utilization.** If it's below ~80%, the GPU is starving for data or waiting on the CPU.

---

### 2. Operator View

Breaks down time at the PyTorch operation level (`aten::matmul`, `aten::relu`, etc.).

Sort by **Total Time** or **Self Time**:
- **Total Time** — the op plus all its child operations
- **Self Time** — only the op itself, excluding children

**Key metric: Host Time vs Device Time.** If a specific op has high Host Time, the CPU is spending too long dispatching it before the GPU ever sees it.

---

### 3. GPU Kernel View

Shows the actual CUDA kernels on the hardware (`volta_sgemm_128x64_nn`, etc.) rather than PyTorch function names.

Sort by **Total Duration** to find the dominant kernels.

**Key metric: Tensor Cores Used.** If your heavy compute kernels (matmul, conv) show `No` in the Tensor Core column, you are leaving significant performance on the table. Modern NVIDIA GPUs can do FP16/BF16 matrix ops an order of magnitude faster with Tensor Cores.

---

### 4. Trace View

The most detailed view — a chronological Chrome-tracing-style timeline. CPU threads at the top, GPU streams at the bottom.

**Key metric: Gaps and Overlaps.** The GPU stream should be a continuous dense band of color. White space means the GPU is idle.

```
CPU: [dispatch][dispatch][dispatch]
GPU:    [kernel][kernel]            [kernel]
                          ↑ gap = GPU waiting on CPU
```

---

### 5. Distributed View (DDP / Multi-Node)

Shows communication overhead from NCCL operations (`nccl:all_reduce`, `nccl:broadcast`).

**Key metric: Computation vs Communication Overlap.** Ideally, gradient communication happens while the next backward pass is computing. If computation stops entirely while NCCL runs, you have a communication bottleneck.

```
Ideal:
  Backward: [====backward====][===backward===]
  NCCL:                  [==all_reduce==]

Bottleneck:
  Backward: [====backward====]
  NCCL:                       [==all_reduce==]  ← serial
```

---

## Part 2: Bottlenecks and Fixes

### Bottleneck 1 — DataLoader (Data-Bound)

The GPU is faster than the CPU can feed it.

**How to spot:**
- Overview: high DataLoader percentage
- Trace: large gaps on the GPU timeline between steps while the CPU is busy

**Fixes:**

```python
DataLoader(
    dataset,
    num_workers=8,      # match your CPU core count
    pin_memory=True,    # faster H2D transfer via pinned memory
    persistent_workers=True,
)
```

- Move data transforms to the GPU where possible
- Store data on NVMe, not spinning disk
- Use `torchvision.transforms.v2` for faster CPU transforms

---

### Bottleneck 2 — CPU Overhead (CPU-Bound)

Data is ready, but the CPU is too slow dispatching ops or synchronizing with the GPU.

**How to spot:**
- Overview: low GPU utilization, high CPU time
- Trace: tiny fragmented GPU kernels with small gaps, busy CPU thread

**Fixes:**

```python
# Bad — .item() forces CPU/GPU sync every step
for step, batch in enumerate(loader):
    loss = model(batch).sum()
    running_loss += loss.item()   # sync point

# Good — accumulate the tensor, sync periodically
for step, batch in enumerate(loader):
    loss = model(batch).sum()
    running_loss += loss          # no sync
    if step % 100 == 0:
        print(running_loss.item())  # sync only here
```

```python
# torch.compile fuses ops and reduces kernel launches (PyTorch 2.0+)
model = torch.compile(model)
```

Avoid `.item()`, `.tolist()`, `.cpu()` inside the hot loop — each is a synchronization barrier.

---

### Bottleneck 3 — Pure Compute / Memory Bandwidth (GPU-Bound)

GPU utilization is near 100% but the step is still slow.

**How to spot:**
- Overview: GPU utilization ~100%
- GPU Kernel: dominant kernels show `No` under Tensor Cores

**Fixes:**

```python
# Automatic Mixed Precision — halves memory, enables Tensor Cores
from torch.amp import autocast, GradScaler

scaler = GradScaler()

with autocast(device_type="cuda", dtype=torch.float16):
    loss = model(batch)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

```python
# FlashAttention via SDPA (PyTorch 2.0+) — fused attention kernel
out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

If you have free VRAM, increasing batch size makes matmuls more efficient (better GPU utilization per kernel launch).

---

### Bottleneck 4 — Communication Overhead (Distributed-Bound)

GPUs spend more time exchanging gradients than computing.

**How to spot:**
- Distributed View: high `all_reduce` time
- Trace: long NCCL blocks with no overlapping compute

**Fixes:**

```python
# Gradient accumulation — reduce communication frequency
accumulation_steps = 4

optimizer.zero_grad()
for i, batch in enumerate(loader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()          # one all_reduce per N steps
        optimizer.zero_grad()
```

```python
# Tune DDP bucket size for your network bandwidth
model = torch.nn.parallel.DistributedDataParallel(
    model,
    bucket_cap_mb=25,   # default is 25 MB; increase for high-bandwidth links
)
```

---

## Quick Reference

| Symptom | View to open | Likely cause | First fix |
|---------|-------------|--------------|-----------|
| GPU util < 80% | Overview → Trace | DataLoader or CPU overhead | `num_workers`, `pin_memory` |
| Fragmented GPU kernels | Trace | CPU dispatch overhead | `torch.compile`, remove `.item()` |
| Tensor Cores = No | GPU Kernel | FP32 compute | Enable AMP (`autocast`) |
| High NCCL time | Distributed | Gradient sync bottleneck | Gradient accumulation, tune `bucket_cap_mb` |
| H2D transfer in hot path | Trace | Data not pre-loaded to GPU | `pin_memory`, prefetch |
