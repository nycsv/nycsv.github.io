---
title: torchrun
description: Multi-node distributed training with torchrun — core concepts and best practices
tags: [pytorch, distributed, training, torchrun]
---

## What torchrun Does

`torchrun` is PyTorch's launcher for distributed training. It replaces the older `torch.distributed.launch` and handles:

- Spawning one process per GPU
- Setting distributed environment variables (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`, etc.)
- Fault tolerance via automatic worker restarts (up to `--max-restarts`)
- Rendezvous — coordinating process group initialization across nodes

Each process runs the **same script** and uses its rank to determine what data/shard it owns.

---

## Key Environment Variables

| Variable | Meaning |
|---|---|
| `RANK` | Global rank of this process (0 … WORLD_SIZE-1) |
| `LOCAL_RANK` | Rank within this node (0 … nproc_per_node-1) |
| `WORLD_SIZE` | Total number of processes across all nodes |
| `MASTER_ADDR` | Hostname/IP of rank-0 node |
| `MASTER_PORT` | Port for the rendezvous |
| `LOCAL_WORLD_SIZE` | Number of processes on this node |

---

## Single-Node Launch

```bash
torchrun \
  --standalone \          # shortcut: sets up rendezvous on localhost
  --nproc-per-node=4 \    # one process per GPU
  train.py
```

`--standalone` is equivalent to `--rdzv-backend=c10d --rdzv-endpoint=localhost:PORT --nnodes=1`.

---

## Multi-Node Launch

Run this **on every node**. Only `--node-rank` differs.

```bash
# Node 0 (master)
torchrun \
  --nnodes=2 \
  --nproc-per-node=4 \
  --rdzv-id=job42 \
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

> `--rdzv-id` must be the same string on all nodes. It namespaces the rendezvous so multiple jobs can share the same endpoint.

---

## Elastic / Dynamic Node Count

```bash
torchrun \
  --nnodes=2:4 \          # min:max — job starts at 2, scales up to 4
  --nproc-per-node=4 \
  --max-restarts=3 \
  --rdzv-id=job42 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=<MASTER_IP>:29500 \
  train.py
```

Workers that fail or are added trigger a **re-rendezvous**. The script must handle re-initialization gracefully.

---

## Minimal Training Script Pattern

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    dist.init_process_group(backend="nccl")  # torchrun sets env vars automatically

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    model = MyModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Only rank 0 should log / save checkpoints
    if dist.get_rank() == 0:
        print(f"World size: {dist.get_world_size()}")

    # ... training loop ...

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

---

## Best Practices

### Data Loading
```python
sampler = DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, sampler=sampler, batch_size=batch_per_gpu)

for epoch in range(epochs):
    sampler.set_epoch(epoch)  # required for proper shuffling per epoch
    for batch in loader:
        ...
```

Each rank sees a non-overlapping shard. Without `set_epoch`, all epochs see the same shuffle order.

### Checkpointing
```python
if dist.get_rank() == 0:
    torch.save({
        "model": model.module.state_dict(),   # unwrap DDP with .module
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }, "checkpoint.pt")

dist.barrier()  # all ranks wait before continuing
```

- Save only on rank 0 to avoid write conflicts
- `dist.barrier()` after save ensures all ranks resume together

### Gradient Synchronization
DDP all-reduces gradients automatically after `loss.backward()`. To skip sync for gradient accumulation:

```python
with model.no_sync():   # suppress all-reduce for N-1 steps
    loss.backward()
loss.backward()         # sync on the last step
optimizer.step()
```

### find_unused_parameters
```python
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
```

Only enable if your model has conditional forward paths (dynamic computation graphs). It adds overhead on every step.

### NCCL Tuning
```bash
export NCCL_IB_DISABLE=0         # enable InfiniBand if available
export NCCL_DEBUG=INFO            # verbose logging for debugging hangs
export NCCL_SOCKET_IFNAME=eth0    # specify network interface
export NCCL_TIMEOUT=1800          # seconds before collective op timeout
```

---

## Common Failure Modes

| Symptom | Likely Cause |
|---|---|
| Hangs at `init_process_group` | Firewall blocking `MASTER_PORT`, wrong `MASTER_ADDR` |
| `NCCL error: unhandled system error` | NIC not found, try `NCCL_SOCKET_IFNAME` |
| Out-of-sync loss across ranks | `sampler.set_epoch()` missing |
| Deadlock after checkpoint | Missing `dist.barrier()` after rank-0 save |
| Gradient NaN after accumulation | `no_sync()` not used, gradients accumulated then double all-reduced |

---

## Rendezvous Backends

| Backend | Use Case |
|---|---|
| `c10d` (default) | Most cases; no external dependency |
| `etcd` / `etcd-v2` | Elastic jobs, production Kubernetes |
| `zeus` | Internal Meta use, not public |

For elastic training on Kubernetes, prefer `etcd` — it survives pod restarts, while `c10d` requires the master endpoint to stay alive.
