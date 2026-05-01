---
title: Perfetto Trace Debugging for Distributed Training
description: How to read Perfetto traces to diagnose GPU idle gaps, AllReduce overlap, straggler ranks, and DataLoader stalls in distributed PyTorch training
tags: [pytorch, distributed, profiling, perfetto, debugging]
date: 2026-04-23
---

Perfetto is the recommended viewer for PyTorch profiler traces — the same `.json` files that TensorBoard's Trace tab shows, but with a faster, more flexible UI.

Open your trace at **[ui.perfetto.dev](https://ui.perfetto.dev)** → drag & drop the `.pt.trace.json` file.

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

Each rank writes its own trace file. For distributed runs, you'll have one file per rank.

---

## UI Layout

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

**Zoom**: scroll wheel. **Pan**: click-drag. **Click a slice**: details in the bottom panel.

---

## What to Look for

### GPU Idle Gaps

```
CUDA stream:  [kernel▓▓▓][  gap  ][kernel▓▓▓]
```

White space on the CUDA stream = GPU waiting. Click the gap — the tooltip shows the CPU op that caused it.

Common causes:
- CPU kernel dispatch too slow (many small ops)
- DataLoader workers not keeping up
- NCCL AllReduce blocking the next forward pass

---

### AllReduce Overlap

Good — AllReduce runs concurrently with backward:
```
CUDA compute:  [backward▓▓▓▓▓▓▓▓▓]
NCCL stream:          [AllReduce▓▓▓▓]   ← starts mid-backward
```

Bad — AllReduce serialized after backward:
```
CUDA compute:  [backward▓▓▓▓▓▓▓]
NCCL stream:                    [AllReduce▓▓▓▓▓▓]
```

If you see no overlap:
- Bucket size too large → `DDP(model, bucket_cap_mb=25)` or smaller
- `find_unused_parameters=True` disables the overlap
- Very small models simply have nothing to overlap

---

### Straggler Rank

Open multiple rank traces (File → Open multiple) and align by timestamp:

```
Rank 0:  [fwd▓▓][bwd▓▓▓][AllReduce waiting...]
Rank 1:  [fwd▓▓][bwd▓▓▓▓▓▓▓▓▓▓▓▓▓]→[AllReduce]
                          ^ straggler
```

All ranks block at AllReduce until the slowest one arrives.

Causes:
- Uneven data shards → `DistributedSampler` + `drop_last=True`
- Thermal throttle on one GPU → `nvidia-smi -q -d PERFORMANCE`
- Uneven `find_unused_parameters` overhead

---

### DataLoader Stall

```
CPU thread:    [DataLoader.__next__ ████████████]   ← long blocking
CUDA stream:   [                   idle          ][kernel]
```

Fixes:
- More `num_workers`
- `pin_memory=True` + `non_blocking=True` on `.to(device)`
- Move preprocessing offline (pre-tokenize, pre-normalize)

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `W` / `S` | Zoom in / out |
| `A` / `D` | Pan left / right |
| `F` | Fit selection to screen |
| `M` | Mark a region |
| `/` | Search by slice name |
| `Shift+click` | Select a time range → shows duration |

---

## Common Patterns and Fixes

| Trace Pattern | Diagnosis | Fix |
|---|---|---|
| GPU idle after backward | AllReduce not overlapping | Reduce `bucket_cap_mb`, remove `find_unused_parameters` |
| One rank always last at AllReduce | Straggler | `drop_last=True`, check thermal throttle |
| Long DataLoader slice each step | I/O bound | More workers, `pin_memory`, prefetch |
| Tiny dense kernels, low throughput | Launch overhead | `torch.compile()`, fuse ops |
| Long `Memcpy HtoD` | Unpinned memory | `pin_memory=True` |
| All ranks idle simultaneously | Load imbalance in forward | `record_shapes=True`, profile per layer |
