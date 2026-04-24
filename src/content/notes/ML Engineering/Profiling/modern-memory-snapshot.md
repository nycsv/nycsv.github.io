---
title: Modern Memory Snapshot in PyTorch
description: How to capture, visualize, and analyze CUDA memory snapshots in PyTorch 2.1+ to debug OOM errors and memory leaks
tags: [pytorch, cuda, memory, profiling, debugging]
date: 2026-04-23
---

## What It Is

Modern Memory Snapshot is PyTorch 2.1+'s official CUDA memory tracking tool built around `torch.cuda.memory._snapshot()`. Unlike the older numeric APIs (`memory_allocated()`, `memory_reserved()`), it records **every allocation and deallocation event with a full call stack** — so you can see exactly which line of code is holding onto memory and why.

```python
# Old way — numbers only, no location
print(torch.cuda.memory_allocated())  # 1.2 GB — but where?

# Modern Snapshot — allocation site + call stack per tensor
torch.cuda.memory._record_memory_history()
# ... run code ...
snapshot = torch.cuda.memory._snapshot()
# Each tensor → filename, line number, size, live/freed status
```

---

## Basic Usage

```python
import torch
import pickle

device = "cuda"

# 1. Start recording (max_entries caps the event buffer)
torch.cuda.memory._record_memory_history(max_entries=100_000)

# 2. Run your model
model = torch.nn.Linear(4096, 4096).to(device)
optimizer = torch.optim.Adam(model.parameters())

for step in range(5):
    x = torch.randn(64, 4096, device=device)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# 3. Capture snapshot
snapshot = torch.cuda.memory._snapshot()

# 4. Save to disk
with open("memory_snapshot.pkl", "wb") as f:
    pickle.dump(snapshot, f)

# 5. Stop recording
torch.cuda.memory._record_memory_history(enabled=None)
```

Visualize in browser:

```bash
python -m torch.cuda._memory_viz trace memory_snapshot.pkl -o snapshot.html
```

---

## Detecting a Memory Leak

A common leak pattern: appending the loss tensor (which keeps the computation graph alive) instead of calling `.item()`.

```python
import torch
import pickle

torch.cuda.memory._record_memory_history(max_entries=100_000)

model = torch.nn.TransformerEncoder(
    torch.nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
    num_layers=6,
).cuda()
optimizer = torch.optim.AdamW(model.parameters())

loss_history = []  # bug: retains the full computation graph each step

for step in range(20):
    x = torch.randn(32, 128, 512, device="cuda")
    loss = model(x).mean()

    loss_history.append(loss)        # graph accumulates → leak
    # loss_history.append(loss.item())  # fix: detach to scalar

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"step {step:02d} | allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

snapshot = torch.cuda.memory._snapshot()
with open("leak_snapshot.pkl", "wb") as f:
    pickle.dump(snapshot, f)

torch.cuda.memory._record_memory_history(enabled=None)
```

Memory climbs every step:

```
step 00 | allocated: 312.4 MB
step 05 | allocated: 698.1 MB
step 10 | allocated: 1082.3 MB
step 19 | allocated: 1842.3 MB
```

Opening the snapshot HTML shows the exact line holding each unreleased tensor.

---

## Before / After Comparison

Take two snapshots bracketing only the region of interest, after warming up so initial allocations don't pollute the diff.

```python
import torch
import pickle
import gc

def save_snapshot(path: str) -> None:
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
    x = torch.randn(32, 2048, device="cuda")
    model(x).sum().backward()
    optimizer.step()
    optimizer.zero_grad()

save_snapshot("snap_after.pkl")    # measurement point

torch.cuda.memory._record_memory_history(enabled=None)
```

```bash
python -m torch.cuda._memory_viz trace snap_before.pkl -o before.html
python -m torch.cuda._memory_viz trace snap_after.pkl  -o after.html
```

Compare the two HTML files to isolate what grew between the two points.

---

## Capturing on OOM

Wrap your training loop to dump a snapshot automatically when CUDA runs out of memory — the snapshot captures the state right before the crash.

```python
import torch
import pickle

torch.cuda.memory._record_memory_history(max_entries=100_000)

try:
    model = torch.nn.Linear(4096, 4096).cuda()
    for step in range(1000):
        x = torch.randn(512, 4096, device="cuda")  # intentionally large
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

```bash
python -m torch.cuda._memory_viz trace oom_snapshot.pkl -o oom.html
```

The timeline will show exactly which allocation pushed the device over the limit.

---

## Distributed Training (torchrun)

Each rank has its own GPU, so save a separate snapshot per rank.

```python
import torch
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

If one rank OOMs while others are fine, its snapshot will show the diverging allocation pattern.

---

## Analyzing the Snapshot Directly

```python
import pickle

with open("memory_snapshot.pkl", "rb") as f:
    snapshot = pickle.load(f)

# Top-level keys
# snapshot["segments"]      — list of CUDA memory segments
# snapshot["device_traces"] — full allocation/free event timeline

# Find the largest live tensors and their source location
blocks = [
    block
    for seg in snapshot["segments"]
    for block in seg["blocks"]
    if block["state"] == "active_allocated"
]
blocks.sort(key=lambda b: b["size"], reverse=True)

print("Top 5 largest live tensors:")
for b in blocks[:5]:
    size_mb = b["size"] / 1e6
    frames = b.get("frames", [])
    loc = f"{frames[0]['filename']}:{frames[0]['line']}" if frames else "unknown"
    print(f"  {size_mb:7.1f} MB  ←  {loc}")
```

Example output:

```
Top 5 largest live tensors:
  256.0 MB  ←  train.py:47            # optimizer state
  128.0 MB  ←  train.py:52            # gradient buffer
   64.0 MB  ←  model/attention.py:88  # attention scores
   32.0 MB  ←  train.py:44            # input batch
   16.0 MB  ←  model/ffn.py:31        # intermediate activations
```

---

## API Reference

| API | Purpose |
|-----|---------|
| `_record_memory_history(max_entries=N)` | Start recording; N caps the event buffer |
| `_record_memory_history(enabled=None)` | Stop recording |
| `_snapshot()` | Return snapshot dict at the current moment |
| `memory_viz trace snap.pkl -o out.html` | Generate interactive timeline HTML |
| `snapshot["segments"]` | Per-segment block states (live / freed) |
| `snapshot["device_traces"]` | Full chronological alloc/free event log |

---

## When to Use It

| Symptom | Action |
|---------|--------|
| `memory_allocated()` grows every step | Take before/after snapshots across one iteration |
| OOM with no obvious cause | Wrap loop in `try/except OutOfMemoryError`, dump on crash |
| One DDP rank OOMs, others don't | Compare per-rank snapshots |
| Gradient accumulation bloating memory | Check live tensors between `.zero_grad()` calls |

---

## Combining with `torch.profiler.profile`

The memory snapshot API and `torch.profiler.profile` are complementary. The profiler adds operator-level CPU/CUDA timelines on top of memory tracking when you enable `profile_memory=True`, `record_shapes=True`, and `with_stack=True`.

```python
import torch
import pickle

torch.cuda.memory._record_memory_history(max_entries=100_000)

model = torch.nn.Linear(4096, 4096).cuda()
optimizer = torch.optim.Adam(model.parameters())

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    profile_memory=True,   # per-op memory delta in the Chrome trace
    record_shapes=True,    # tensor shapes alongside op names
    with_stack=True,       # Python call stack per op
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb_logs"),
) as prof:
    for step in range(5):
        x = torch.randn(64, 4096, device="cuda")
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prof.step()

# Save memory snapshot alongside the Chrome trace
snapshot = torch.cuda.memory._snapshot()
with open("snapshot_with_profiler.pkl", "wb") as f:
    pickle.dump(snapshot, f)

torch.cuda.memory._record_memory_history(enabled=None)
```

```bash
# Memory snapshot timeline
python -m torch.cuda._memory_viz trace snapshot_with_profiler.pkl -o memory.html

# Operator timeline (open in Chrome → chrome://tracing or TensorBoard)
tensorboard --logdir ./tb_logs
```

The two views are complementary: the profiler shows **what ops ran and how long**, the snapshot shows **which allocations survived across steps**.

---

## Step-Range Scheduling

Instead of recording the full run, activate memory history only for the steps you care about — this matches the NeMo `PytorchProfilerCallback` pattern of `start_step` / `end_step`.

```python
import torch
import pickle

model = torch.nn.Linear(4096, 4096).cuda()
optimizer = torch.optim.Adam(model.parameters())

PROFILE_START = 10   # skip noisy warm-up steps
PROFILE_END   = 15

for step in range(20):
    x = torch.randn(64, 4096, device="cuda")
    loss = model(x).sum()
    loss.backward()
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

Skipping warm-up keeps the snapshot clean — one-time weight and optimizer-state allocations don't appear as false positives.

---

## `ExecutionTraceObserver` (Chakra Traces)

NeMo pairs `ExecutionTraceObserver` with `torch.profiler.profile` to export Chakra host traces — operator-level execution graphs used for workload replay and roofline analysis. Add it alongside memory snapshot to get all three views from one run.

```python
import torch
import pickle
from pathlib import Path

trace_dir = Path("traces")
(trace_dir / "host").mkdir(parents=True, exist_ok=True)
(trace_dir / "device").mkdir(parents=True, exist_ok=True)

# Chakra host trace (operator execution graph)
observer = torch.profiler.ExecutionTraceObserver()
observer.register_callback(str(trace_dir / "host" / "rank-0.json"))

torch.cuda.memory._record_memory_history(max_entries=100_000)

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=0, warmup=2, active=3),
    on_trace_ready=lambda prof: prof.export_chrome_trace(
        str(trace_dir / "device" / "rank-0.json")
    ),
    execution_trace_observer=observer,   # attach Chakra observer
    profile_memory=True,
    with_stack=True,
) as prof:
    model = torch.nn.Linear(4096, 4096).cuda()
    optimizer = torch.optim.Adam(model.parameters())

    for step in range(10):
        model(torch.randn(64, 4096, device="cuda")).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        prof.step()

snapshot = torch.cuda.memory._snapshot()
with open(trace_dir / "memory_snapshot.pkl", "wb") as f:
    pickle.dump(snapshot, f)

torch.cuda.memory._record_memory_history(enabled=None)

try:
    observer.unregister_callback()
except RuntimeError:
    pass  # already unregistered by profiler stop
```

Output layout:

```
traces/
├── host/rank-0.json      ← Chakra execution graph (operator DAG)
├── device/rank-0.json    ← Kineto Chrome trace (CPU+CUDA timeline)
└── memory_snapshot.pkl   ← CUDA memory allocation history
```

- `host/rank-0.json` — load into [Chakra](https://github.com/mlcommons/chakra) for workload replay
- `device/rank-0.json` — open in `chrome://tracing` or Perfetto
- `memory_snapshot.pkl` — visualize with `memory_viz trace`

---

## Reusable Lightning Callback

Wrapping all three into a Lightning callback (modeled after NeMo's `PytorchProfilerCallback`) makes it drop-in for any training script.

```python
import pickle
from pathlib import Path
from typing import Any, Optional

import torch
from lightning.pytorch.callbacks import Callback


class MemoryProfilerCallback(Callback):
    """
    Combines torch.profiler.profile + memory snapshot + ExecutionTraceObserver
    for a specific step range. Saves all artifacts to trace_dir.
    """

    def __init__(
        self,
        start_step: int,
        end_step: int,
        trace_dir: str = "traces",
        rank: int = 0,
    ):
        if end_step < start_step:
            raise ValueError("end_step must be >= start_step")

        self.start_step = start_step
        self.end_step = end_step
        self.rank = rank

        self.trace_dir = Path(trace_dir)
        (self.trace_dir / "host").mkdir(parents=True, exist_ok=True)
        (self.trace_dir / "device").mkdir(parents=True, exist_ok=True)

        self._observer = torch.profiler.ExecutionTraceObserver()
        self._profiler: Optional[torch.profiler.profile] = None
        self._active = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.global_step != self.start_step:
            return

        self._observer.register_callback(
            str(self.trace_dir / "host" / f"rank-{self.rank}.json")
        )
        torch.cuda.memory._record_memory_history(max_entries=100_000)

        self._profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            profile_memory=True,
            with_stack=True,
            on_trace_ready=lambda p: p.export_chrome_trace(
                str(self.trace_dir / "device" / f"rank-{self.rank}.json")
            ),
            execution_trace_observer=self._observer,
        )
        self._profiler.start()
        self._active = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self._active:
            return

        if trainer.global_step < self.end_step:
            self._profiler.step()
        else:
            self._stop()

    def _stop(self):
        self._profiler.stop()
        self._active = False

        snapshot = torch.cuda.memory._snapshot()
        with open(self.trace_dir / f"memory_rank{self.rank}.pkl", "wb") as f:
            pickle.dump(snapshot, f)
        torch.cuda.memory._record_memory_history(enabled=None)

        try:
            self._observer.unregister_callback()
        except RuntimeError:
            pass
```

Usage:

```python
import lightning as L

trainer = L.Trainer(
    max_epochs=3,
    callbacks=[
        MemoryProfilerCallback(start_step=10, end_step=15, trace_dir="traces"),
    ],
)
trainer.fit(model, datamodule)
```
