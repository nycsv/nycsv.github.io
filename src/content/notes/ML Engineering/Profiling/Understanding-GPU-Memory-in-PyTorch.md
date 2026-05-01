---
title: Understanding GPU Memory in PyTorch
description: Where GPU memory actually goes during training — parameters, gradients, optimizer states, activations — and how to measure each
tags: [pytorch, cuda, memory, profiling, training]
date: 2026-04-23
---

When you hit an OOM, "reduce batch size" is the blunt fix. Knowing *what* is actually taking up memory lets you be smarter about it.

---

## Where the Memory Goes

For a model with `P` parameters in FP32:

| Component | Size | Notes |
|---|---|---|
| Parameters | `4P` bytes | the model weights |
| Gradients | `4P` bytes | one grad tensor per parameter |
| Optimizer states (Adam) | `8P` bytes | momentum + variance, both FP32 |
| Activations | varies | depends on batch size and sequence length |
| **Total (Adam, FP32)** | **≥ 16P bytes** | before activations |

A 7B parameter model at FP32 needs at least **112 GB** just for the optimizer state — that's before a single batch goes through.

---

## Mixed Precision Changes the Picture

With BF16/FP16 training (`torch.autocast` + `GradScaler`):

| Component | Size |
|---|---|
| Parameters (BF16) | `2P` bytes |
| Master weights (FP32 copy) | `4P` bytes |
| Gradients (BF16) | `2P` bytes |
| Optimizer states (FP32) | `8P` bytes |
| **Total** | **~16P bytes** |

Mixed precision saves activation memory (BF16 activations are half the size) but optimizer states are still FP32 — the saving is mostly in activations and gradient communication.

---

## Measuring What's Actually Allocated

```python
import torch

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters())

# Before forward pass
print(f"After model init:  {torch.cuda.memory_allocated() / 1e9:.2f} GB")

x = get_batch().cuda()
loss = model(x).mean()
print(f"After forward:     {torch.cuda.memory_allocated() / 1e9:.2f} GB")

loss.backward()
print(f"After backward:    {torch.cuda.memory_allocated() / 1e9:.2f} GB")

optimizer.step()
optimizer.zero_grad()
print(f"After optimizer:   {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

The jump from "after model init" to "after forward" is your activation memory for that batch size. The jump from "after forward" to "after backward" is gradient memory.

---

## Activation Memory Is the Variable One

Activations are intermediate tensors saved during the forward pass for use in the backward pass. Their size scales with:
- batch size
- sequence length (quadratic for attention — `O(seq_len²)`)
- number of layers

**Gradient checkpointing** trades compute for memory by discarding activations and recomputing them during backward:

```python
from torch.utils.checkpoint import checkpoint

# Instead of: out = layer(x)
out = checkpoint(layer, x, use_reentrant=False)
```

This roughly halves activation memory at the cost of ~33% more compute.

---

## The Reserved vs Allocated Gap

PyTorch's allocator doesn't return memory to CUDA after each free — it holds onto it in a pool for reuse. So:

```python
torch.cuda.memory_allocated()  # memory in active use
torch.cuda.memory_reserved()   # memory held by the allocator (allocated + cached free blocks)
```

`reserved - allocated` is memory the allocator is holding but not currently using. If `reserved` is near your GPU limit but `allocated` is much lower, **fragmentation** is the problem — not peak usage.

```python
torch.cuda.empty_cache()  # release the cached free blocks back to CUDA
```

Call this after validation or between major phases if fragmentation is hurting you.

---

## Quick Cheatsheet: Memory Reduction Techniques

| Technique | Memory saved | Cost |
|---|---|---|
| BF16/FP16 training | ~50% activations | Slight precision risk |
| Gradient checkpointing | ~50% activations | ~33% more compute |
| Gradient accumulation | — (no saving) | Reduces batch-per-GPU, not peak memory |
| `zero_grad(set_to_none=True)` | gradient buffer freed | Minor |
| FSDP / DeepSpeed ZeRO | optimizer states sharded | Distributed setup required |
| Flash Attention (`F.scaled_dot_product_attention`) | `O(seq_len)` vs `O(seq_len²)` | Near-free with PyTorch 2.0+ |

---

## Further Reading

- [Visualizing All Allocations over Time](https://pytorch.org/blog/understanding-gpu-memory-1/) — PyTorch blog, walks through the memory snapshot tool
- [Training Large Models: Memory](https://huggingface.co/blog/train_memory) — Hugging Face guide covering FSDP, ZeRO, and mixed precision
- [OOMs with FSDP, torchao, and debugging](https://www.youtube.com/watch?v=UvRl4ansfCg) — PyTorch dev talk with live debugging walkthrough
