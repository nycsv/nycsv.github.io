---
title: NeMo PytorchProfilerCallback — Chakra Traces and Execution Profiling
description: How NeMo's PytorchProfilerCallback works, what it captures, and how to adapt the pattern for custom distributed training pipelines
tags: [pytorch, nemo, profiling, chakra, distributed, lightning]
date: 2026-04-23
---

## What It Does

NeMo's `PytorchProfilerCallback` (introduced in NeMo 2.x) is a Lightning callback that coordinates three profiling tools for a bounded step range:

| Tool | Output | What it shows |
|------|--------|---------------|
| `torch.profiler.profile` | `device/rank-N.json` | CPU + CUDA op timeline (Kineto) |
| `ExecutionTraceObserver` | `host/rank-N.json` | Operator execution graph (Chakra) |
| `torch.profiler.schedule` | automatic | warmup → active → stop lifecycle |

The key design: profiling activates at `start_step` and stops at `end_step`, so it doesn't pay overhead for the full run and skips noisy warm-up allocations.

---

## Source Walkthrough

```python
# nemo/lightning/pytorch/callbacks/pytorch_profiler.py (v2.7.0)

class PytorchProfilerCallback(Callback, IOMixin):
    def __init__(
        self,
        start_step: int,
        end_step: int,
        warmup_steps: int = 0,
        active_steps: int = 1,
        trace_dir: str = None,
        profiler_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Two separate output dirs for host vs device traces
        self.chakra_host_trace_path   = self.trace_dir / "host"
        self.chakra_device_trace_path = self.trace_dir / "device"

        self.trace_observer = torch.profiler.ExecutionTraceObserver()

        self.profiler = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=self.warmup_steps,
                active=self.active_steps,
            ),
            on_trace_ready=lambda prof: prof.export_chrome_trace(
                str(self.chakra_device_trace_path / f"rank-{rank}.json")
            ),
            execution_trace_observer=self.trace_observer,
            # profiler_kwargs can override/extend any of the above
        )

    def on_train_batch_start(self, trainer, ...):
        if trainer.global_step == self.start_step:
            self.trace_observer.register_callback(host_trace_path)
            self.profiler.start()

    def on_train_batch_end(self, trainer, ...):
        if self.is_profiling:
            if trainer.global_step < self.end_step:
                self.profiler.step()   # advance the schedule
            else:
                self._stop_profiler()  # flush + export

    def _stop_profiler(self):
        self.profiler.stop()
        self.trace_observer.unregister_callback()
```

Three things worth noting:

1. **`IOMixin`** — NeMo's serialization mixin; lets the callback config be saved/restored with the checkpoint.
2. **`profiler_kwargs`** — merges into the base dict, so callers can add `profile_memory=True` or `record_shapes=True` without subclassing.
3. **Guard against double-start** — `on_train_batch_start` checks `self.is_profiling` before calling `profiler.start()` to prevent duplicate profiler instances under resumed training.

---

## Output Files

```
traces/
├── host/
│   ├── rank-0.json    ← Chakra execution trace (operator DAG)
│   ├── rank-1.json
│   └── ...
└── device/
    ├── rank-0.json    ← Kineto Chrome trace (CPU + CUDA timeline)
    ├── rank-1.json
    └── ...
```

**`device/rank-N.json`** — open in `chrome://tracing` or [ui.perfetto.dev](https://ui.perfetto.dev):
- GPU kernel durations and launch gaps
- CPU-GPU synchronization points
- Memory copy (H2D / D2H) events

**`host/rank-N.json`** — load into [Chakra](https://github.com/mlcommons/chakra):
- Operator-level execution DAG
- Used for workload characterization and replay
- Input to roofline analysis tooling

---

## Minimal Reproduction (no NeMo)

```python
import torch
from pathlib import Path
from lightning.pytorch.callbacks import Callback


class PytorchProfilerCallback(Callback):
    def __init__(
        self,
        start_step: int,
        end_step: int,
        warmup_steps: int = 0,
        active_steps: int = 1,
        trace_dir: str = "traces",
        profiler_kwargs: dict = None,
    ):
        if end_step < start_step:
            raise ValueError("end_step must be >= start_step")

        self.start_step = start_step
        self.end_step = end_step

        self.trace_dir = Path(trace_dir)
        self.host_dir = self.trace_dir / "host"
        self.device_dir = self.trace_dir / "device"
        self.host_dir.mkdir(parents=True, exist_ok=True)
        self.device_dir.mkdir(parents=True, exist_ok=True)

        self.trace_observer = torch.profiler.ExecutionTraceObserver()

        base_kwargs = dict(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=0, warmup=warmup_steps, active=active_steps
            ),
            on_trace_ready=self._export_device_trace,
            execution_trace_observer=self.trace_observer,
        )
        if profiler_kwargs:
            base_kwargs.update(profiler_kwargs)

        self.profiler = torch.profiler.profile(**base_kwargs)
        self.is_profiling = False

    def _rank(self):
        import torch.distributed as dist
        return dist.get_rank() if dist.is_initialized() else 0

    def _export_device_trace(self, prof):
        path = self.device_dir / f"rank-{self._rank()}.json"
        prof.export_chrome_trace(str(path))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.global_step != self.start_step or self.is_profiling:
            return

        host_path = self.host_dir / f"rank-{self._rank()}.json"
        self.trace_observer.register_callback(str(host_path))

        self.profiler.start()
        self.is_profiling = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.is_profiling:
            return

        if trainer.global_step < self.end_step:
            self.profiler.step()
        else:
            self._stop()

    def _stop(self):
        self.profiler.stop()
        self.is_profiling = False
        try:
            self.trace_observer.unregister_callback()
        except RuntimeError:
            pass
```

---

## Adding Memory Snapshot

NeMo's callback doesn't include `_record_memory_history` — add it to get the full picture:

```python
import pickle

class MemoryAwarePytorchProfilerCallback(PytorchProfilerCallback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.global_step == self.start_step:
            torch.cuda.memory._record_memory_history(max_entries=100_000)
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def _stop(self):
        super()._stop()

        snap = torch.cuda.memory._snapshot()
        path = self.trace_dir / f"memory_rank{self._rank()}.pkl"
        with open(path, "wb") as f:
            pickle.dump(snap, f)

        torch.cuda.memory._record_memory_history(enabled=None)
```

Now one callback produces all three artifact types per rank:

```
traces/
├── host/rank-0.json        ← Chakra operator DAG
├── device/rank-0.json      ← Kineto Chrome trace
└── memory_rank0.pkl        ← CUDA memory snapshot
```

---

## Step-Range Scheduling in Practice

The `warmup_steps` parameter inside `torch.profiler.schedule` is distinct from the step offset controlled by `start_step`:

```
global steps:  0  1  2  3  4  5  6  7  8  9  10 11 12 ...
                              ^start_step=5
                              |--warmup=2--|--active=3--|
                                           ^profiler captures here
```

- `start_step` → when `profiler.start()` is called (Lightning level)
- `warmup` → profiler internal warm-up before it begins capturing (Kineto level)
- `active` → how many steps are actually recorded to the trace

Setting `warmup_steps > 0` lets the profiler's internal CUDA event queues stabilize before committing data to disk.

---

## Usage

```python
import lightning as L

trainer = L.Trainer(
    max_epochs=3,
    callbacks=[
        MemoryAwarePytorchProfilerCallback(
            start_step=10,
            end_step=20,
            warmup_steps=2,
            active_steps=5,
            trace_dir="./traces",
            profiler_kwargs={
                "profile_memory": True,
                "record_shapes": True,
                "with_stack": True,
            },
        )
    ],
)
trainer.fit(model, datamodule)
```

```bash
# Visualize device trace
# Open traces/device/rank-0.json in chrome://tracing or ui.perfetto.dev

# Visualize memory snapshot
python -m torch.cuda._memory_viz trace traces/memory_rank0.pkl -o memory.html
open memory.html
```

---

## Comparison: NeMo vs Plain `torch.profiler`

| | NeMo `PytorchProfilerCallback` | Plain `torch.profiler.profile` |
|---|---|---|
| Activation | `start_step` / `end_step` | manual `with` block |
| Chakra host trace | yes (`ExecutionTraceObserver`) | no |
| Memory snapshot | no (add via subclass) | no (add manually) |
| Multi-rank | per-rank file naming | manual |
| `profiler_kwargs` override | yes | n/a |
| Resume safety | `is_profiling` guard | n/a |
