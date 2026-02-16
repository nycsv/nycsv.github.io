---
title: "Triton devlog: first steps"
date: 2026-01-16
tags: ["triton", "asr", "serving"]
translationKey: "triton-devlog-001"
draft: false
---

Today I organized the basic model loading and I/O path for a Triton Python backend.

## Checklist
- Create a minimal repro for errors
- Decompose bottlenecks from a serving perspective
- Define targets using p50/p95 latency
