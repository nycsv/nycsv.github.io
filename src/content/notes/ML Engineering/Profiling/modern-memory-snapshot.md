---
title: Modern Memory Snapshot in PyTorch
description: How to capture, visualize, and analyze CUDA memory snapshots in PyTorch 2.1+ to debug OOM errors and memory leaks
tags: [pytorch, cuda, memory, profiling, debugging]
date: 2026-04-23
---

The older `torch.cuda.memory_allocated()` tells you *how much* memory is in use, but not *where*. The modern memory snapshot API records every allocation and deallocation with a full call stack — so when something holds onto memory unexpectedly, you can see the exact line of code responsible.

```python
# Old way — a number with no location
print(torch.cuda.memory_allocated())  # 1.2 GB — but which tensor? from where?

# Modern snapshot — allocation site + call stack per tensor
torch.cuda.memory._record_memory_history()
# ... run your code ...
snapshot = torch.cuda.memory._snapshot()
# Each entry: filename, line number, tensor size, live/freed status
```

---

## Basic Usage

```python
import torch
import pickle

torch.cuda.memory._record_memory_history(max_entries=100_000)

model = torch.nn.Linear(4096, 4096).cuda()
optimizer = torch.optim.Adam(model.parameters())

for step in range(5):
    x = torch.randn(64, 4096, device="cuda")
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

snapshot = torch.cuda.memory._snapshot()

with open("memory_snapshot.pkl", "wb") as f:
    pickle.dump(snapshot, f)

torch.cuda.memory._record_memory_history(enabled=None)
```

Visualize in browser:

```bash
python -m torch.cuda._memory_viz trace memory_snapshot.pkl -o snapshot.html
open snapshot.html
```

---

## Detecting a Memory Leak

A common pattern: appending the loss tensor to a list instead of calling `.item()`. The list holds a reference to the computation graph, which keeps growing.

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)

model = torch.nn.TransformerEncoder(
    torch.nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
    num_layers=6,
).cuda()
optimizer = torch.optim.AdamW(model.parameters())

loss_history = []

for step in range(20):
    x = torch.randn(32, 128, 512, device="cuda")
    loss = model(x).mean()

    loss_history.append(loss)          # leaks: retains the full graph each step
    # loss_history.append(loss.item()) # fix: detach to a Python scalar

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"step {step:02d} | allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
```

Memory climbs every step:

```
step 00 | allocated: 312.4 MB
step 05 | allocated: 698.1 MB
step 19 | allocated: 1842.3 MB
```

The snapshot HTML will highlight the exact line holding each unreleased tensor.

---

## Before / After Comparison

Take two snapshots bracketing the region of interest, after warming up so one-time allocations don't appear as false positives.

```python
import torch, pickle, gc

def save_snapshot(path):
    with open(path, "wb") as f:
        pickle.dump(torch.cuda.memory._snapshot(), f)

torch.cuda.memory._record_memory_history(max_entries=100_000)

model = torch.nn.Linear(2048, 2048).cuda()
optimizer = torch.optim.Adam(model.parameters())

# warm-up: let one-time allocations settle
for _ in range(3):
    model(torch.randn(32, 2048, device="cuda")).sum().backward()
    optimizer.step()
    optimizer.zero_grad()

gc.collect()
torch.cuda.empty_cache()
save_snapshot("snap_before.pkl")   # baseline

for step in range(10):
    model(torch.randn(32, 2048, device="cuda")).sum().backward()
    optimizer.step()
    optimizer.zero_grad()

save_snapshot("snap_after.pkl")    # measurement point

torch.cuda.memory._record_memory_history(enabled=None)
```

```bash
python -m torch.cuda._memory_viz trace snap_before.pkl -o before.html
python -m torch.cuda._memory_viz trace snap_after.pkl  -o after.html
```

Anything that grew between the two snapshots is a candidate for the leak.

---

## Capturing on OOM

Wrap your training loop to dump a snapshot automatically at the moment of the OOM — the snapshot captures state right before the crash.

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)

try:
    model = torch.nn.Linear(4096, 4096).cuda()
    for step in range(1000):
        x = torch.randn(512, 4096, device="cuda")
        model(x).sum().backward()
except torch.cuda.OutOfMemoryError:
    snapshot = torch.cuda.memory._snapshot()
    with open("oom_snapshot.pkl", "wb") as f:
        pickle.dump(snapshot, f)
    print("OOM snapshot saved → oom_snapshot.pkl")
    raise
finally:
    torch.cuda.memory._record_memory_history(enabled=None)
```

The timeline will show exactly which allocation pushed the device over the limit.

---

## Distributed Training

Each rank has its own GPU, so save a separate snapshot per rank.

```python
import torch.distributed as dist
import pickle

dist.init_process_group("nccl")
rank = dist.get_rank()

torch.cuda.memory._record_memory_history(max_entries=100_000)

model = torch.nn.parallel.DistributedDataParallel(
    torch.nn.Linear(4096, 4096).cuda()
)

for step in range(5):
    model(torch.randn(64, 4096, device="cuda")).sum().backward()

snapshot = torch.cuda.memory._snapshot()
with open(f"snapshot_rank{rank}.pkl", "wb") as f:
    pickle.dump(snapshot, f)

torch.cuda.memory._record_memory_history(enabled=None)
```

```bash
for rank in 0 1 2 3; do
    python -m torch.cuda._memory_viz trace snapshot_rank${rank}.pkl \
        -o snapshot_rank${rank}.html
done
```

If one rank OOMs while others don't, its snapshot will show the diverging allocation pattern.

---

## Inspecting the Snapshot Directly

```python
import pickle

with open("memory_snapshot.pkl", "rb") as f:
    snapshot = pickle.load(f)

# snapshot["segments"]      — CUDA memory segments
# snapshot["device_traces"] — full alloc/free event timeline

# Find the largest live tensors and their source location
blocks = [
    b
    for seg in snapshot["segments"]
    for b in seg["blocks"]
    if b["state"] == "active_allocated"
]
blocks.sort(key=lambda b: b["size"], reverse=True)

for b in blocks[:5]:
    size_mb = b["size"] / 1e6
    frames = b.get("frames", [])
    loc = f"{frames[0]['filename']}:{frames[0]['line']}" if frames else "unknown"
    print(f"  {size_mb:7.1f} MB  ←  {loc}")
```

Example output:

```
  256.0 MB  ←  train.py:47            # optimizer state
  128.0 MB  ←  train.py:52            # gradient buffer
   64.0 MB  ←  model/attention.py:88  # attention scores
```

---

## Step-Range Scheduling

Record only the steps you care about — skipping warm-up keeps the snapshot clean.

```python
PROFILE_START = 10
PROFILE_END   = 15

for step in range(20):
    x = torch.randn(64, 4096, device="cuda")
    model(x).sum().backward()
    optimizer.step()
    optimizer.zero_grad()

    if step == PROFILE_START:
        torch.cuda.memory._record_memory_history(max_entries=100_000)

    if step == PROFILE_END:
        snapshot = torch.cuda.memory._snapshot()
        with open(f"snapshot_step{step}.pkl", "wb") as f:
            pickle.dump(snapshot, f)
        torch.cuda.memory._record_memory_history(enabled=None)
        break
```

---

## API Reference

| API | Purpose |
|---|---|
| `_record_memory_history(max_entries=N)` | Start recording; N caps the event buffer |
| `_record_memory_history(enabled=None)` | Stop recording |
| `_snapshot()` | Capture current snapshot as a dict |
| `memory_viz trace snap.pkl -o out.html` | Generate interactive timeline HTML |
| `snapshot["segments"]` | Per-segment block states (live / freed) |
| `snapshot["device_traces"]` | Chronological alloc/free event log |

---

## When to Use It

| Symptom | Action |
|---|---|
| `memory_allocated()` grows every step | Before/after snapshots across one iteration |
| OOM with no obvious cause | Wrap loop in `try/except OutOfMemoryError`, dump on crash |
| One DDP rank OOMs, others don't | Compare per-rank snapshots |
| Gradient accumulation bloating memory | Check live tensors between `.zero_grad()` calls |
