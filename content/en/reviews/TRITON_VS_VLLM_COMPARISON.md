---
title: "qwen3_asr_triton vs speechLLM (vLLM): Performance & Architecture Comparison"
date: 2026-03-02
tags: ["ASR", "Qwen3-ASR", "vLLM", "Triton", "streaming", "benchmark", "KV-cache"]
---

# qwen3_asr_triton vs speechLLM (vLLM): Comparison

---

## Functional Equivalence

Both implementations are **functionally equivalent** at the protocol level:

| Aspect | qwen3_asr_triton | speechLLM (vLLM) |
|---|---|---|
| WebSocket protocol | ✅ start / binary PCM / end | ✅ start / binary PCM / end |
| Token rollback | ✅ re-infer last 2 chunks | ✅ re-infer last 2 chunks |
| Text stability | ✅ LCP-style committed/partial | ✅ LCP committed/partial |
| Context reset | ✅ 300s + inject `last_committed` | ✅ 300s + inject `last_committed` |
| Inference backend | HuggingFace Transformers (direct) | vLLM (scheduler-managed) |
| Serving layer | Triton Python Backend | FastAPI + Uvicorn |

---

## KV Cache Behavior

This is a **critical architectural difference**.

### LLM Decoder KV Cache

**Neither implementation maintains a KV cache across chunks.**

Both call fresh generation per chunk:

```
Chunk N arrives:
  1. Re-encode ALL accumulated audio (0..N) → embeddings
  2. Full decoder prefill from scratch (no past_key_values)
  3. Generate max_new_tokens new tokens
  4. Rollback last unfixed_chunk_num chunks' tokens
```

The decoder always restarts from an empty KV cache on every chunk. This is by design because the transcript grows and the previous decode state is stale — the audio embeddings change (Phase 1) or the prefix changes (token rollback), so cached KV states would be invalid anyway.

### Audio Encoder KV Cache (Encoder Output Caching)

| Phase | Triton | vLLM (speechLLM) | Description |
|---|---|---|---|
| Phase 1 (current) | ✅ implemented | ✅ implemented | Full accumulated audio re-encoded every chunk — O(N²) |
| Phase 2 (incremental) | 🚧 test_phase2.py (planned) | ❌ not planned | Cache per-chunk encoder outputs, reuse → O(N) |

**The Triton implementation has a planned Phase 2** that would cache encoder outputs per-chunk in `session.encoder_output_cache` and pass `inputs_embeds` to skip audio_tower on subsequent chunks. This exploits Qwen3-ASR's windowed attention property (100-frame training / 400-frame inference windows — past chunk encoder outputs don't change when new chunks arrive).

**The vLLM implementation has no equivalent** because vLLM's model execution path doesn't expose a hook to inject pre-computed encoder embeddings at the per-chunk level without custom model registration.

---

## Benchmark Results

### Single Session, 640ms chunks

| Audio Duration | Metric | Transformers/Triton | vLLM (speechLLM) | Winner |
|---|---|---|---|---|
| 10s | TTFT | ~0.18s | ~0.12s | vLLM (marginal) |
| 10s | RTF | ~0.31x | ~0.28x | vLLM (marginal) |
| 20s | TTFT | ~0.21s | ~0.40s | **Transformers 1.9x** |
| 20s | RTF | ~0.36x | ~0.41x | **Transformers** |
| 60s | TTFT | 0.224s | 0.561s | **Transformers 2.5x** |
| 60s | Total | 24.84s | 31.68s | **Transformers 1.28x** |
| 60s | RTF | 0.414x | 0.528x | **Transformers** |

> RTF (Real-Time Factor) = total_inference_time / audio_duration. Lower is better; <1.0 means faster than real-time.

### Concurrency

| Sessions | Transformers/Triton | vLLM (speechLLM) |
|---|---|---|
| c=1 | Works, RTF ~0.41x | Works, RTF ~0.53x |
| c=4 | Works, RTF ~1.1x | Works, RTF ~1.3x |
| c=8 | Works, RTF ~2.0x | **Fails** (scheduler overwhelmed) |

---

## Why Transformers Wins at Scale

Both use **Phase 1 (O(N²) re-encoding)** — each new chunk re-encodes all audio accumulated so far. But the overhead sources differ:

- **Transformers**: ~300ms constant overhead per chunk (raw forward pass, no scheduler)
- **vLLM**: chunk inference time grows linearly with session length due to PagedAttention scheduler overhead and KV cache management for the growing audio context

vLLM's scheduler is optimized for batched LLM decode throughput, not for high-frequency small-batch audio encoder workloads. At 640ms chunk intervals, the scheduler overhead dominates.

```
Per-chunk latency growth (approximate):
  Transformers: latency ≈ C₀ + α × audio_length     (α ≈ small)
  vLLM:         latency ≈ C₀ + β × audio_length     (β > α due to scheduler)
```

At 60s audio (≈94 chunks of 640ms), this difference compounds to a 1.28x total time advantage.

---

## Summary: Which is Better?

| Scenario | Winner | Reason |
|---|---|---|
| Short audio (<10s) | **vLLM** (marginal) | Lower cold-start, vLLM batch warmup helps |
| Long audio (>20s) | **Transformers/Triton** | No scheduler overhead growth |
| High concurrency (c≥8) | **Transformers/Triton** | vLLM scheduler breaks down |
| Phase 2 (future) | **Transformers/Triton** | Incremental encoder caching already prototyped |
| LLM decode throughput | **vLLM** | PagedAttention for multi-turn batching |
| Operational simplicity | **vLLM (speechLLM)** | Single process, no Triton ensemble |
| Multi-GPU horizontal scaling | **Dynamo E/PD** | NIXL RDMA encoder→decoder separation |

**Bottom line**: For streaming ASR specifically, `qwen3_asr_triton` (Transformers) is faster and more scalable under load. vLLM's core strengths (PagedAttention, continuous batching) apply to LLM decode token generation — they are largely irrelevant overhead for a repeated small-batch audio encoding workload.

---

## Architecture Diagram

```
qwen3_asr_triton (Transformers/Triton)
────────────────────────────────────────
Client
  │ WebSocket (PCM16LE + JSON)
  ▼
FastAPI :8003 (or Triton gRPC :8001)
  │
  ▼
Per-Session State Machine (SessionState)
  ├── buffer: raw PCM bytes
  ├── audio_accum: float32 np.array (grows with each chunk)
  ├── text: transcript so far
  ├── prompt_raw: fixed system prompt
  └── (NO KV cache stored)
  │
  ▼ _run_inference() — every 640ms chunk
  │
  ├── processor(text=prompt, audio=audio_accum)   ← FULL accumulated audio
  │   └── audio_tower (encoder) — O(N²) total
  │
  └── model.generate(**inputs, max_new_tokens=32)  ← fresh decode, no past_key_values
      └── Qwen3ForCausalLM decoder (full prefill)


speechLLM (vLLM)
────────────────────────────────────────
Client
  │ WebSocket (PCM16LE + JSON)
  ▼
FastAPI :8004 (Uvicorn)
  │
  ▼
Per-Session State Machine
  ├── streaming_state: qwen_asr library state
  ├── last_committed: stable prefix
  └── audio_accum_samples: counter
  │
  ▼ streaming_transcribe() — every 320ms chunk (via asyncio.to_thread)
  │
  └── vLLM AsyncLLM scheduler
      ├── audio_tower (encoder) — O(N²) total, + scheduler overhead
      └── Qwen3ForCausalLM decoder — PagedAttention KV (per-request, not cross-chunk)
```

---

## File References

| File | Description |
|---|---|
| `/home/eesung/sourcecode/qwen3_asr_triton/triton/model_repository/qwen3_asr_streaming/1/model.py` | Triton Python Backend — Phase 1 streaming |
| `/home/eesung/sourcecode/qwen3_asr_triton/server/qwen3_asr_server.py` | FastAPI wrapper over same Transformers model |
| `/home/eesung/sourcecode/qwen3_asr_triton/server/test_phase2.py` | Planned Phase 2 incremental encoder caching |
| `/home/eesung/sourcecode/qwen3_asr_triton/TECHNICAL_COMPARISON.md` | Triton project's own benchmark documentation |
| `/home/eesung/sourcecode/qwen3_asr_triton/server/benchmark_comparison.py` | Benchmark runner: Transformers vs vLLM |
| `/home/eesung/sourcecode/production_asr/speechLLM/qwen_asr_server.py` | vLLM-based streaming server (speechLLM) |
