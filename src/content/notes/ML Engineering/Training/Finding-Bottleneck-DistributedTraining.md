---
title: Finding the Bottleneck in Distributed Training
description: How to capture PyTorch profiler traces and read them in Perfetto to identify GPU idle gaps, AllReduce overlap issues, straggler ranks, and DataLoader stalls
tags: [pytorch, distributed, profiling, perfetto, debugging]
date: 2026-04-23
---

When distributed training is slower than expected, the bottleneck usually falls into one of four categories: GPU idle time, AllReduce serialization, a straggler rank, or DataLoader stalls. A Perfetto trace makes each of these visible.

---

## Capturing a Trace

```python
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=tensorboard_trace_handler("./trace_dir"),
    record_shapes=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(loader):
        train_step(batch)
        prof.step()
```

Each rank writes its own trace file. Open them at **ui.perfetto.dev** → drag & drop the `.json` or `.pt.trace.json` file.

---

## Perfetto UI Layout

```
Timeline (top)
├── Process: rank 0
│   ├── Thread: CPU ops
│   │   └── [forward] [backward] [optimizer]
│   ├── Thread: CUDA stream 0
│   │   └── [kernel] [kernel] [kernel] ...
│   ├── Thread: NCCL stream
│   │   └── [AllReduce] [AllReduce] ...
│   └── Thread: DataLoader workers
└── Process: rank 1
    └── ...
```

**Zoom**: scroll wheel. **Pan**: click-drag. **Select a slice**: click → details in the bottom panel.

---

## What to Look for

### 1. GPU Idle Gaps (Bubbles)

```
CUDA stream:  [kernel▓▓▓][  gap  ][kernel▓▓▓]
```

A white gap means the GPU is waiting. Click the gap — the tooltip shows the CPU op that caused it.

Causes:
- CPU-bound kernel launch (many small ops)
- DataLoader workers not keeping up
- AllReduce blocking the next forward pass

---

### 2. AllReduce Overlap

Ideal — DDP overlaps AllReduce with the backward pass:
```
CUDA compute:  [backward▓▓▓▓▓▓▓▓▓]
NCCL stream:          [AllReduce▓▓▓▓]   ← starts mid-backward
```

Bad — AllReduce runs after backward finishes:
```
CUDA compute:  [backward▓▓▓▓▓▓▓]
NCCL stream:                    [AllReduce▓▓▓▓▓▓]   ← gap between
```

If you see no overlap:
- Bucket size too large → try `DDP(model, bucket_cap_mb=25)` (or smaller)
- `find_unused_parameters=True` disables the overlap optimization
- Very small models have no backward phases long enough to overlap

---

### 3. Straggler Rank

Load multiple rank traces simultaneously (File → Open multiple). Align by timestamp.

```
Rank 0:  [fwd▓▓][bwd▓▓▓][AllReduce waiting...]
Rank 1:  [fwd▓▓][bwd▓▓▓▓▓▓▓▓▓▓▓▓▓]→[AllReduce]
                          ^ straggler
```

All ranks block at AllReduce until the slowest finishes. Causes:
- Uneven data shard sizes → `DistributedSampler` + `drop_last=True`
- Thermal throttling on one GPU → check `nvidia-smi -q -d PERFORMANCE`
- Uneven `find_unused_parameters` overhead

---

### 4. DataLoader Stall

```
CPU thread:    [DataLoader.__next__ ████████████]   ← long blocking call
CUDA stream:   [                   idle          ][kernel]
```

Fixes:
- Increase `num_workers`
- `pin_memory=True` + `non_blocking=True` on `.to(device)`
- Move preprocessing offline (pre-tokenize, pre-normalize)

---

### 5. Too Many Small Kernels (Launch Overhead)

```
CUDA stream: [k][k][k][k][k][k][k]   ← tiny kernels, lots of gaps between them
```

Each kernel launch costs ~5–20 µs on the CPU. If kernels are shorter than that, you're launch-bound.

Fixes:
- `torch.compile()` — fuses ops into fewer kernels
- `torch.cuda.amp.autocast()` — reduces precision, often merges ops

---

### 6. Long Memcpy (Unpinned Memory)

```
CUDA stream: [Memcpy HtoD ████████][kernel]
```

Fixes:
- `DataLoader(pin_memory=True)` — locks CPU memory for async DMA transfer
- `tensor.to(device, non_blocking=True)` — overlaps copy with compute

---

## Workflow

1. Open trace → zoom into one full training step (forward + backward + optimizer)
2. Is the CUDA stream dense or full of gaps?
3. Find the longest gap → click it → read the CPU op name
4. Check the NCCL stream — is AllReduce overlapping backward?
5. Open rank traces side by side → look for straggler at AllReduce boundaries
6. Is the DataLoader thread the longest CPU span per step?

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `W` / `S` | Zoom in / out |
| `A` / `D` | Pan left / right |
| `F` | Fit selection to screen |
| `M` | Mark / highlight a region |
| `/` | Search by slice name |
| `Shift+click` | Select time range → shows duration |

---

## Quick Reference

| Trace Pattern | Diagnosis | Fix |
|---|---|---|
| GPU idle after backward | AllReduce not overlapping | Reduce `bucket_cap_mb`, remove `find_unused_parameters` |
| One rank always last at AllReduce | Straggler | `drop_last=True`, check thermal throttle |
| Long DataLoader slice each step | I/O bound | More workers, `pin_memory`, prefetch |
| Tiny dense kernels, low throughput | Launch overhead | `torch.compile()`, fuse ops |
| Long `Memcpy HtoD` | Unpinned memory | `pin_memory=True` |
| All ranks idle simultaneously | Load imbalance in forward | Profile per-layer with `record_shapes=True` |
