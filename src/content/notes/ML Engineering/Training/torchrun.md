---
title: Distributed Training with torchrun
description: How to launch single-node and multi-node distributed training with torchrun, including environment variables, DDP patterns, and common failure modes
tags: [pytorch, distributed, training, torchrun, ddp]
date: 2026-04-23
---

`torchrun` is PyTorch's launcher for distributed training. It replaces the older `torch.distributed.launch` and handles process spawning, environment setup, and fault tolerance automatically.

Each process runs the **same script**. Processes use their rank to figure out which data shard they own and whether to log or save checkpoints.

---

## Key Environment Variables

`torchrun` sets these automatically — your script just reads them.

| Variable | Meaning |
|---|---|
| `RANK` | Global rank (0 … WORLD_SIZE-1) |
| `LOCAL_RANK` | Rank within this node (0 … nproc_per_node-1) |
| `WORLD_SIZE` | Total processes across all nodes |
| `MASTER_ADDR` | Hostname/IP of rank-0 node |
| `MASTER_PORT` | Port for rendezvous |

---

## Single-Node Launch

```bash
torchrun \
  --standalone \          # sets up rendezvous on localhost automatically
  --nproc-per-node=4 \    # one process per GPU
  train.py
```

`--standalone` is shorthand for `--rdzv-backend=c10d --rdzv-endpoint=localhost:PORT --nnodes=1`.

---

## Multi-Node Launch

Run this on **every node** — only `--node-rank` differs.

```bash
# Node 0 (master)
torchrun \
  --nnodes=2 \
  --nproc-per-node=4 \
  --rdzv-id=job42 \          # must match on all nodes
  --rdzv-backend=c10d \
  --rdzv-endpoint=<MASTER_IP>:29500 \
  --node-rank=0 \
  train.py

# Node 1
torchrun \
  --nnodes=2 \
  --nproc-per-node=4 \
  --rdzv-id=job42 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=<MASTER_IP>:29500 \
  --node-rank=1 \
  train.py
```

---

## Elastic / Dynamic Node Count

```bash
torchrun \
  --nnodes=2:4 \       # min:max — starts at 2 nodes, can scale to 4
  --nproc-per-node=4 \
  --max-restarts=3 \
  --rdzv-id=job42 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=<MASTER_IP>:29500 \
  train.py
```

Worker failures or additions trigger a **re-rendezvous**. Your script needs to handle re-initialization cleanly (load from last checkpoint, reinitialize process group).

---

## Minimal Training Script

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    # torchrun has already set RANK, LOCAL_RANK, WORLD_SIZE
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model = MyModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    if dist.get_rank() == 0:
        print(f"World size: {dist.get_world_size()}")

    # ... training loop ...

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

---

## Common Patterns

### Data loading

```python
sampler = DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, sampler=sampler, batch_size=batch_per_gpu)

for epoch in range(epochs):
    sampler.set_epoch(epoch)  # without this, all epochs get the same shuffle
    for batch in loader:
        ...
```

### Checkpointing

```python
if dist.get_rank() == 0:
    torch.save({
        "model": model.module.state_dict(),  # .module unwraps DDP
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }, "checkpoint.pt")

dist.barrier()  # all ranks wait here before continuing
```

### Gradient accumulation without double all-reduce

```python
for i, batch in enumerate(loader):
    # suppress all-reduce for the first N-1 micro-steps
    with model.no_sync() if (i + 1) % accum_steps != 0 else contextlib.nullcontext():
        loss = model(batch) / accum_steps
        loss.backward()

    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### `find_unused_parameters`

```python
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
```

Only enable this if your model has conditional forward paths. It adds per-step overhead and disables the AllReduce/backward overlap optimization.

---

## NCCL Tuning

```bash
export NCCL_IB_DISABLE=0          # enable InfiniBand if available
export NCCL_DEBUG=INFO             # verbose logging — useful for debugging hangs
export NCCL_SOCKET_IFNAME=eth0    # specify the network interface
export NCCL_TIMEOUT=1800          # seconds before a collective op times out
```

---

## Common Failure Modes

| Symptom | Likely Cause |
|---|---|
| Hangs at `init_process_group` | Firewall blocking `MASTER_PORT`, wrong `MASTER_ADDR` |
| `NCCL error: unhandled system error` | NIC not found — try setting `NCCL_SOCKET_IFNAME` |
| Out-of-sync loss across ranks | `sampler.set_epoch()` missing |
| Deadlock after checkpoint save | Missing `dist.barrier()` after rank-0 save |
| Gradient NaN after accumulation | `no_sync()` not used — gradients accumulated then double all-reduced |

---

## Rendezvous Backends

| Backend | Use Case |
|---|---|
| `c10d` (default) | Standard use, no external dependencies |
| `etcd` / `etcd-v2` | Elastic jobs, Kubernetes — survives pod restarts |
