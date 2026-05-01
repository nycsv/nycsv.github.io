---
title: Merging NeMo FSDP Distributed Checkpoints into a Single File
description: How to consolidate PyTorch DCP (distributed checkpoint) shards from NeMo FSDP training into one .ckpt file, including the model config stored in meta.pt
tags: [pytorch, fsdp, nemo, checkpoint, distributed]
date: 2026-05-01
---

After FSDP training with NeMo 2.x, your checkpoint directory looks like this:

```
checkpoint_dir/
├── .metadata
├── meta.pt          # training config / model hyperparams
├── __0_0.distcp     # shard for rank 0
├── __1_0.distcp
├── ...
└── __7_0.distcp     # shard for rank 7
```

These are **PyTorch DCP (Distributed Checkpoint)** shards — one per rank. To use the weights outside of a distributed context (e.g., inference, fine-tuning on a single GPU), you need to consolidate them into a single file.

---

## The Script

```python
# merge_fsdp_ckpt.py
#
# Merges NeMo FSDP distributed checkpoint shards into a single .ckpt file.
# Requires PyTorch >= 2.2 for dcp_to_torch_save.
# No GPU or distributed setup needed — runs entirely on CPU.

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from pathlib import Path
import argparse


def merge(ckpt_dir: str, output_path: str):
    ckpt_dir = Path(ckpt_dir)
    output_path = Path(output_path)

    # Load the training config saved by NeMo alongside the shards.
    # meta.pt typically contains model hyperparams, tokenizer config, etc.
    meta = torch.load(ckpt_dir / "meta.pt", map_location="cpu", weights_only=False)
    cfg = meta.get("cfg", meta)

    # dcp_to_torch_save reads all .distcp shards and writes a single torch-save file.
    # It handles the shard-merging logic internally, no ranks needed.
    dcp_to_torch_save(str(ckpt_dir), str(output_path))

    # Re-open the merged file and attach cfg so everything lives in one place.
    raw = torch.load(output_path, map_location="cpu", weights_only=False)

    # Normalize: some DCP saves already wrap weights under "state_dict", some don't.
    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw

    merged = {
        "state_dict": state_dict,
        "cfg": cfg,
    }
    torch.save(merged, output_path)

    print(f"Saved → {output_path}")
    print(f"state_dict keys (top-level): {list(state_dict.keys())[:5]} ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_dir", help="Path to the distcp checkpoint directory")
    parser.add_argument("output",   help="Output .ckpt file path")
    args = parser.parse_args()
    merge(args.ckpt_dir, args.output)
```

**Run it:**

```bash
python merge_fsdp_ckpt.py /path/to/checkpoint_dir output.ckpt
```

---

## What's Happening Under the Hood

`dcp_to_torch_save` reads `.metadata` to understand the full tensor layout, then reconstructs each tensor by concatenating the relevant slices from each `.distcp` shard. The result is equivalent to what you'd get if you'd saved with `torch.save` from a single-GPU run.

---

## Things to Watch Out For

**Memory** — The entire model is loaded into RAM at once. Make sure you have at least 2× the model size in free memory before running this.

**`weights_only=False`** — Required because `meta.pt` contains non-tensor Python objects (OmegaConf / dataclass configs). If your meta only has tensors, you can flip this to `True` for safety.

**NeMo 1.x** — Older NeMo uses a different checkpoint format (`.nemo` zip files). This script targets NeMo 2.x FSDP2 checkpoints only.
