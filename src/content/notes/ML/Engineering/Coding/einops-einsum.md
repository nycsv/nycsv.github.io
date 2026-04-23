---
title: "Einops & Einsum"
date: 2026-04-01
tags: ["ml", "einops", "einsum", "tensor"]
description: "Tensor manipulation patterns using einops and einsum for cleaner ML code."
---

## Composition of axes

`einops.rearrange` lets you describe tensor transformations declaratively.

```python
from einops import rearrange

# (batch, channels, height, width) → (batch, height, width, channels)
x = rearrange(x, 'b c h w -> b h w c')

# Merge batch and sequence dimensions
x = rearrange(x, 'b t c -> (b t) c')
```

The arrow notation makes the shape contract explicit — you never have to guess what a chain of `.view()` and `.permute()` calls does.

## Decomposition of axis

You can split axes by naming them with products:

```python
# Split channels into heads × head_dim
x = rearrange(x, 'b (h d) -> b h d', h=8)

# Unfold spatial grid into patches
patches = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)
```

## Meet einops.reduce

`reduce` combines rearrange semantics with aggregation:

```python
from einops import reduce

# Global average pooling
pooled = reduce(x, 'b c h w -> b c', 'mean')

# Max across time
x = reduce(x, 'b t d -> b d', 'max')
```

This replaces `.mean(dim=[2, 3])` with something self-documenting.

## Einsum

`torch.einsum` describes contractions with index notation:

```python
import torch

# Matrix multiply: (b, i, k) × (b, k, j) → (b, i, j)
out = torch.einsum('bik,bkj->bij', A, B)

# Attention scores: (b, h, q, d) × (b, h, k, d) → (b, h, q, k)
scores = torch.einsum('bhqd,bhkd->bhqk', Q, K)

# Outer product: (n,) × (m,) → (n, m)
outer = torch.einsum('i,j->ij', a, b)
```

The rule: repeated indices are summed over; indices that appear only on one side are kept.

Combine einsum with einops for maximum clarity:

```python
# Multi-head attention in ~5 lines
Q = rearrange(q_proj(x), 'b t (h d) -> b h t d', h=num_heads)
K = rearrange(k_proj(x), 'b t (h d) -> b h t d', h=num_heads)
V = rearrange(v_proj(x), 'b t (h d) -> b h t d', h=num_heads)

scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / math.sqrt(d_head)
attn   = scores.softmax(dim=-1)
out    = torch.einsum('bhqk,bhkd->bhqd', attn, V)
out    = rearrange(out, 'b h t d -> b t (h d)')
```
