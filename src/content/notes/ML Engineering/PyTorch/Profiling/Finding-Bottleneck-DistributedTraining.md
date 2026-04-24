---
title: Finding the Bottleneck in Distributed Training
description: How to read ui.perfetto.dev traces to find bottlenecks in distributed PyTorch training
tags: [pytorch, distributed, profiling, perfetto, debugging]
date: 2026-04-23
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

**Zoom**: scroll wheel. **Pan**: click-drag. **Select a slice**: click → details appear in the bottom panel.

---

## What to Look for — Distributed Bottlenecks

### 1. GPU Idle Gaps (Bubbles)

```
CUDA stream:  [kernel▓▓▓][  gap  ][kernel▓▓▓]
```

A visible white gap on the CUDA stream means the GPU is waiting. Causes:
- **CPU-bound**: CPU kernel launch is too slow (lots of small ops)
- **DataLoader stall**: workers not keeping up
- **NCCL blocking**: AllReduce holding up the next forward pass

> Tip: click the gap — the tooltip shows the preceding CPU op that caused the stall.

---

### 2. AllReduce Duration and Overlap

Ideal (compute-communication overlap with DDP):
```
CUDA compute:  [backward▓▓▓▓▓▓▓▓▓]
NCCL stream:          [AllReduce▓▓▓▓]   ← starts mid-backward
```

Bad (no overlap — AllReduce serialized after backward):
```
CUDA compute:  [backward▓▓▓▓▓▓▓]
NCCL stream:                    [AllReduce▓▓▓▓▓▓]   ← gap between
```

DDP overlaps AllReduce with backward by bucketing gradients. If you see no overlap:
- Bucket size too large → `DDP(model, bucket_cap_mb=25)` (default 25 MB, try smaller)
- `find_unused_parameters=True` disables the overlap optimization
- Single-layer models or very small models have nothing to overlap

---

### 3. Rank Skew (Straggler)

Load multiple rank traces into Perfetto simultaneously (File → Open multiple). Align them by timestamp.

```
Rank 0:  [fwd▓▓][bwd▓▓▓][AllReduce waiting...]
Rank 1:  [fwd▓▓][bwd▓▓▓▓▓▓▓▓▓▓▓▓▓]→[AllReduce]
                          ^ straggler
```

All ranks block at AllReduce until the slowest finishes. Causes:
- Uneven data shard sizes → `DistributedSampler` + `drop_last=True`
- One node has slower GPU or thermal throttling → check `nvidia-smi -q -d PERFORMANCE`
- Uneven `find_unused_parameters` overhead across ranks

---

### 4. DataLoader Stall

```
CPU thread:    [DataLoader.__next__ ████████████]   ← long blocking call
CUDA stream:   [                   idle          ][kernel]
```

The main thread is waiting on a worker. Fix:
- Increase `num_workers`
- Use `pin_memory=True` + `non_blocking=True` on `.to(device)`
- Move preprocessing off the critical path (pre-tokenize, pre-normalize)

---

### 5. CPU Op Fragmentation (Too Many Small Kernels)

```
CUDA stream: [k][k][k][k][k][k][k]   ← many tiny kernels, lots of launch overhead
```

Each kernel launch has ~5–20 µs CPU overhead. If kernels are shorter than that, you're launch-bound. Solutions:
- `torch.compile()` — fuses ops into fewer kernels
- `torch.cuda.amp.autocast()` — reduces precision, often merges ops
- Replace Python loops with batched tensor ops

---

### 6. Memory Copy Overhead

Look for `cudaMemcpy` or `Memcpy HtoD` (Host to Device) slices on the CUDA stream:

```
CUDA stream: [Memcpy HtoD ████████][kernel]
```

Long copies mean data isn't pinned. Fix:
- `DataLoader(pin_memory=True)` — locks CPU memory so DMA transfer is async
- `tensor.to(device, non_blocking=True)` — overlaps copy with compute

---

## Workflow: Finding the Bottleneck

1. **Open trace** → look at one full training step (forward + backward + optimizer)
2. **Check CUDA utilization** — is the GPU stream dense or full of gaps?
3. **Find the longest gap** → click it → read the CPU op name in the details panel
4. **Compare NCCL stream** — is AllReduce overlapping backward or serialized after?
5. **Open rank traces side by side** → look for straggler ranks at AllReduce boundaries
6. **Check DataLoader thread** — is it the longest CPU span per step?

---

## Useful Keyboard Shortcuts (Perfetto UI)

| Key | Action |
|---|---|
| `W` / `S` | Zoom in / out |
| `A` / `D` | Pan left / right |
| `F` | Fit selection to screen |
| `M` | Mark / highlight a region |
| `/` | Search by slice name |
| `Shift+click` | Select time range → shows duration |

---

## Common Patterns and Fixes

| Trace Pattern | Diagnosis | Fix |
|---|---|---|
| GPU idle after backward | AllReduce not overlapping | Reduce `bucket_cap_mb`, remove `find_unused_parameters` |
| One rank always last at AllReduce | Straggler node | `drop_last=True`, check thermal throttle |
| Long DataLoader slice each step | I/O bound | More workers, `pin_memory`, prefetch |
| Tiny dense kernels, low throughput | Launch overhead | `torch.compile()`, fuse ops |
| Long `cudaMemcpy HtoD` | Unpinned memory | `pin_memory=True` |
| All ranks idle simultaneously | Load imbalance in forward | Profile per-layer with `record_shapes=True` |
