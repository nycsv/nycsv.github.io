---
title: NeMo PytorchProfilerCallback — Chakra Traces and Execution Profiling
description: How NeMo's PytorchProfilerCallback works, what it captures, and how to reproduce or extend the pattern in custom training pipelines
tags: [pytorch, nemo, profiling, chakra, distributed, lightning]
date: 2026-04-23
---

NeMo's `PytorchProfilerCallback` (NeMo 2.x) is a Lightning callback that coordinates three profiling tools for a bounded step range, without paying overhead for the full training run.

| Tool | Output | What it shows |
|---|---|---|
| `torch.profiler.profile` | `device/rank-N.json` | CPU + CUDA op timeline (Kineto) |
| `ExecutionTraceObserver` | `host/rank-N.json` | Operator execution graph (Chakra) |
| `torch.profiler.schedule` | automatic | warmup → active → stop lifecycle |

---

## How It Works

```python
# nemo/lightning/pytorch/callbacks/pytorch_profiler.py (v2.7.0)

class PytorchProfilerCallback(Callback, IOMixin):
    def __init__(self, start_step, end_step, warmup_steps=0, active_steps=1,
                 trace_dir=None, profiler_kwargs=None):
        self.trace_observer = torch.profiler.ExecutionTraceObserver()

        self.profiler = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=0, warmup=warmup_steps, active=active_steps,
            ),
            on_trace_ready=lambda prof: prof.export_chrome_trace(
                str(self.chakra_device_trace_path / f"rank-{rank}.json")
            ),
            execution_trace_observer=self.trace_observer,
            # profiler_kwargs merges here — callers can add profile_memory, etc.
        )

    def on_train_batch_start(self, trainer, ...):
        if trainer.global_step == self.start_step:
            self.trace_observer.register_callback(host_trace_path)
            self.profiler.start()

    def on_train_batch_end(self, trainer, ...):
        if self.is_profiling:
            if trainer.global_step < self.end_step:
                self.profiler.step()
            else:
                self._stop_profiler()  # flush + export
```

Three design decisions worth noting:

1. **`IOMixin`** — lets the callback config be saved/restored with the checkpoint
2. **`profiler_kwargs`** — merges into the base dict, so callers can add `profile_memory=True` without subclassing
3. **`is_profiling` guard** — prevents double-start under resumed training

---

## Output Files

```
traces/
├── host/
│   ├── rank-0.json    ← Chakra execution trace (operator DAG)
│   └── rank-1.json
└── device/
    ├── rank-0.json    ← Kineto Chrome trace (CPU + CUDA timeline)
    └── rank-1.json
```

- **`device/rank-N.json`** — open in `chrome://tracing` or [ui.perfetto.dev](https://ui.perfetto.dev)
- **`host/rank-N.json`** — load into [Chakra](https://github.com/mlcommons/chakra) for workload replay and roofline analysis

---

## Step-Range Timing

The `warmup_steps` inside `torch.profiler.schedule` is distinct from `start_step`:

```
global steps:  0  1  2  3  4  5  6  7  8  9  10 11 12
                              ^start_step=5
                              |--warmup=2--|--active=3--|
                                           ^profiler captures here
```

- `start_step` → when `profiler.start()` is called (Lightning level)
- `warmup` → profiler's internal warm-up before writing data to disk (Kineto level)
- `active` → how many steps are actually recorded

Setting `warmup_steps > 0` lets the profiler's CUDA event queues stabilize before committing data.

---

## Minimal Reproduction (no NeMo)

```python
import torch
from pathlib import Path
from lightning.pytorch.callbacks import Callback


class PytorchProfilerCallback(Callback):
    def __init__(self, start_step, end_step, warmup_steps=0, active_steps=1,
                 trace_dir="traces", profiler_kwargs=None):
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
        prof.export_chrome_trace(str(self.device_dir / f"rank-{self._rank()}.json"))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.global_step != self.start_step or self.is_profiling:
            return
        self.trace_observer.register_callback(
            str(self.host_dir / f"rank-{self._rank()}.json")
        )
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

NeMo's callback doesn't include `_record_memory_history`. Extend it to get all three artifact types in one run:

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
        with open(self.trace_dir / f"memory_rank{self._rank()}.pkl", "wb") as f:
            pickle.dump(snap, f)
        torch.cuda.memory._record_memory_history(enabled=None)
```

Output layout per rank:

```
traces/
├── host/rank-0.json        ← Chakra operator DAG
├── device/rank-0.json      ← Kineto Chrome trace
└── memory_rank0.pkl        ← CUDA memory snapshot
```

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
# Device trace
# Open traces/device/rank-0.json in chrome://tracing or ui.perfetto.dev

# Memory snapshot
python -m torch.cuda._memory_viz trace traces/memory_rank0.pkl -o memory.html
```

---

## NeMo vs Plain `torch.profiler`

| | NeMo `PytorchProfilerCallback` | Plain `torch.profiler.profile` |
|---|---|---|
| Activation | `start_step` / `end_step` | manual `with` block |
| Chakra host trace | yes | no |
| Memory snapshot | no (add via subclass) | no (add manually) |
| Multi-rank file naming | automatic | manual |
| `profiler_kwargs` override | yes | n/a |
| Resume safety | `is_profiling` guard | n/a |
